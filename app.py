import json
import logging
import time
import requests
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, field_validator

# Import hàm xử lý logic nghiệp vụ của bạn
from decision_engine import apply_rules

# --- Cấu hình Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("weather_api")

# --- Đường dẫn và Load Metadata ---
ARTIFACT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ARTIFACT_DIR / "models"

with open(ARTIFACT_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

HORIZON = metadata["horizon"]
FEATURES = metadata["features"]

# --- Load Models ---
models_temp = {}
models_rhum = {}
models_prcp = {}

logger.info("Loading model artifacts from %s", MODEL_DIR)
for h in range(1, HORIZON + 1):
    models_temp[h] = joblib.load(MODEL_DIR / f"model_temp_h{h}.joblib")
    models_rhum[h] = joblib.load(MODEL_DIR / f"model_rhum_h{h}.joblib")
    models_prcp[h] = joblib.load(MODEL_DIR / f"model_prcp_h{h}.joblib")
logger.info("Loaded model artifacts successfully for horizon=1..%s", HORIZON)

app = FastAPI(title="Weather Forecast API (Open-Meteo Powered)", version="1.1.0")

# --- Middleware ---
@app.middleware("http")
async def request_timer_middleware(request: Request, call_next):
    started_at = time.perf_counter()
    logger.info("Incoming request: %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.perf_counter() - started_at
        logger.exception("Request failed | elapsed=%.3fs", elapsed)
        raise
    elapsed = time.perf_counter() - started_at
    logger.info("Completed request | status=%s | elapsed=%.3fs", response.status_code, elapsed)
    return response

# --- Models Pydantic ---
class SmartForecastRequest(BaseModel):
    lat: float
    lon: float
    temps: List[float]
    rhums: List[float]

    @field_validator("temps", "rhums")
    @classmethod
    def validate_non_empty(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("List must not be empty")
        return v

# --- Core Functions ---

def fetch_recent_weather_history(lat: float, lon: float, lookback_hours: int = 36) -> pd.DataFrame:
    """
    Lấy dữ liệu thời tiết thực tế gần đây từ Open-Meteo API.
    Nguồn này 'mở' hoàn toàn, không cần API Key và rất tin cậy.
    """
    logger.info("Fetching history from Open-Meteo | lat=%s lon=%s", lat, lon)
    started_at = time.perf_counter()

    # API URL của Open-Meteo
    url = "https://api.open-meteo.com/v1/forecast"
    
    # Request tham số: past_days=2 để đảm bảo đủ dữ liệu cho lags (24h) và rolling windows
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m",
        "past_days": 2, 
        "forecast_days": 1,
        "timezone": "UTC"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        hourly = data["hourly"]
        df = pd.DataFrame({
            "temp": hourly["temperature_2m"],
            "rhum": hourly["relative_humidity_2m"],
            "prcp": hourly["precipitation"],
            "pres": hourly["surface_pressure"],
            "wspd": hourly["wind_speed_10m"]
        }, index=pd.to_datetime(hourly["time"]))
        
        df = df.astype(float)
        
    except Exception as ex:
        logger.exception("Open-Meteo fetch failed")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch weather from Open-Meteo: {str(ex)}"
        )

    elapsed = time.perf_counter() - started_at
    logger.info("Provider returned %s rows in %.3fs", len(df), elapsed)

    # Làm sạch dữ liệu
    df = df.sort_index()
    df = df.interpolate(limit_direction="both").ffill().bfill()
    
    # Cắt lấy số lượng hàng cần thiết cho Feature Engineering (lookback + buffer)
    df = df.tail(lookback_hours + 12).copy()

    if len(df) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough history rows from provider. Got {len(df)}."
        )

    return df

def override_recent_temp_rhum(history: pd.DataFrame, temps: List[float], rhums: List[float]) -> pd.DataFrame:
    if len(temps) != len(rhums):
        raise HTTPException(status_code=400, detail="temps and rhums must have the same length")

    n = len(temps)
    if len(history) < n:
        raise HTTPException(status_code=400, detail="History too short to override")

    out = history.copy()
    # Ghi đè dữ liệu cảm biến thực tế của người dùng vào các dòng cuối cùng
    out.iloc[-n:, out.columns.get_loc("temp")] = temps
    out.iloc[-n:, out.columns.get_loc("rhum")] = rhums
    return out

def build_feature_row_from_history(history_df: pd.DataFrame) -> pd.DataFrame:
    df = history_df.copy()
    
    # Time features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month

    # Cyclic encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 366)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 366)

    use_cols = ["temp", "rhum", "prcp", "wspd", "pres"]
    lags = [1, 2, 3, 6, 12, 24]
    windows = [3, 6, 12, 24]

    for lag in lags:
        for col in use_cols:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    for w in windows:
        df[f"temp_roll_mean_{w}"] = df["temp"].shift(1).rolling(w).mean()
        df[f"temp_roll_std_{w}"] = df["temp"].shift(1).rolling(w).std()
        df[f"rhum_roll_mean_{w}"] = df["rhum"].shift(1).rolling(w).mean()
        df[f"rhum_roll_std_{w}"] = df["rhum"].shift(1).rolling(w).std()
        df[f"prcp_roll_sum_{w}"] = df["prcp"].shift(1).rolling(w).sum()
        df[f"wspd_roll_mean_{w}"] = df["wspd"].shift(1).rolling(w).mean()
        df[f"pres_roll_mean_{w}"] = df["pres"].shift(1).rolling(w).mean()

    df = df.dropna().copy()
    if df.empty:
        raise HTTPException(status_code=400, detail="Feature engineering failed (not enough data)")

    latest = df.iloc[[-1]].copy()
    
    # Kiểm tra xem có thiếu feature nào model yêu cầu không
    missing_features = [f for f in FEATURES if f not in latest.columns]
    if missing_features:
        raise HTTPException(status_code=500, detail={"error": "Missing features", "list": missing_features[:5]})

    return latest[FEATURES]

def generate_forecast_from_feature_row(row: pd.DataFrame, hours: int) -> List[Dict[str, Any]]:
    results = []
    for h in range(1, hours + 1):
        p_temp = float(models_temp[h].predict(row)[0])
        p_rhum = float(models_rhum[h].predict(row)[0])
        p_prcp = max(0.0, float(models_prcp[h].predict(row)[0]))

        results.append({
            "horizon": h,
            "pred_temp": round(p_temp, 2),
            "pred_rhum": round(p_rhum, 2),
            "pred_prcp": round(p_prcp, 2)
        })
    return results

# --- Endpoints ---

@app.get("/")
def root():
    return {
        "status": "online",
        "provider": "Open-Meteo",
        "max_horizon": HORIZON
    }

@app.post("/smart-predict")
def smart_predict(req: SmartForecastRequest, hours: int = Query(default=6, ge=1)):
    if hours > HORIZON:
        raise HTTPException(status_code=400, detail=f"Max hours is {HORIZON}")

    # 1. Lấy dữ liệu lịch sử từ API mở
    history = fetch_recent_weather_history(req.lat, req.lon, lookback_hours=36)
    
    # 2. Ghi đè bằng dữ liệu thực tế từ sensor gửi lên
    history = override_recent_temp_rhum(history, req.temps, req.rhums)

    # 3. Tạo vector đặc trưng (Feature Engineering)
    row = build_feature_row_from_history(history)
    
    # 4. Chạy model dự báo cho n giờ tới
    forecast = generate_forecast_from_feature_row(row, hours)
    
    # 5. Áp dụng rule-based engine để ra quyết định
    decision = apply_rules(forecast)

    return {
        "lat": req.lat,
        "lon": req.lon,
        "forecast": forecast,
        "decision_result": decision
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)