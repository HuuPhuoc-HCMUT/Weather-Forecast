import json
import logging
import time
import requests
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# --- 1. Cấu hình Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("irrigation_ai")

# --- 2. Load Metadata & Models ---
ARTIFACT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ARTIFACT_DIR / "models"

with open(ARTIFACT_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

HORIZON = metadata["horizon"]
FEATURES = metadata["features"]

models_temp = {}
models_rhum = {}
models_prcp = {}

logger.info("Loading model artifacts...")
for h in range(1, HORIZON + 1):
    models_temp[h] = joblib.load(MODEL_DIR / f"model_temp_h{h}.joblib")
    models_rhum[h] = joblib.load(MODEL_DIR / f"model_rhum_h{h}.joblib")
    models_prcp[h] = joblib.load(MODEL_DIR / f"model_prcp_h{h}.joblib")
logger.info("All models loaded successfully.")

# --- 3. Khởi tạo FastAPI & CORS ---
description = """
## 🌿 Hệ thống Tưới cây AI (Smart Irrigation)
API này cung cấp khả năng dự báo thời tiết siêu cục bộ và lập lịch tưới tự động dựa trên độ ẩm đất.
"""

app = FastAPI(title="Smart Irrigation AI", description=description, version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Models Pydantic ---

class AutoIrrigationRequest(BaseModel):
    lat: float = Field(..., example=10.76)
    lon: float = Field(..., example=106.66)
    min_moisture: float = Field(..., description="Ngưỡng dưới độ ẩm đất (%)", example=30.0)
    max_moisture: float = Field(..., description="Ngưỡng trên độ ẩm đất (%)", example=70.0)
    temps: List[float] = Field(..., description="24h nhiệt độ gần nhất")
    hums: List[float] = Field(..., description="24h độ ẩm KK gần nhất")
    moists: List[float] = Field(..., description="24h độ ẩm đất gần nhất")

class SmartForecastRequest(BaseModel):
    lat: float
    lon: float
    temps: List[float]
    rhums: List[float]

# --- 5. Core Weather Functions ---

def fetch_recent_weather_history(lat: float, lon: float, lookback_hours: int = 36) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m",
        "past_days": 2, "forecast_days": 1, "timezone": "UTC"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()["hourly"]
        df = pd.DataFrame({
            "temp": data["temperature_2m"], "rhum": data["relative_humidity_2m"],
            "prcp": data["precipitation"], "pres": data["surface_pressure"], "wspd": data["wind_speed_10m"]
        }, index=pd.to_datetime(data["time"])).astype(float)
        return df.sort_index().interpolate(limit_direction="both").ffill().bfill().tail(lookback_hours + 12)
    except Exception as e:
        logger.error(f"Weather fetch error: {e}")
        raise HTTPException(status_code=502, detail="Weather provider error")

def build_feature_row(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # --- THÊM ĐẦY ĐỦ CÁC CỘT THỜI GIAN Ở ĐÂY ---
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek  # Thêm dòng này
    df["month"] = df.index.month            # Thêm dòng này
    df["day_of_year"] = df.index.dayofyear
    
    # Cyclic encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 366)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 366)

    use_cols = ["temp", "rhum", "prcp", "wspd", "pres"]
    
    # Tạo Lags
    for lag in [1, 2, 3, 6, 12, 24]:
        for col in use_cols:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            
    # Tạo Rolling Windows
    for w in [3, 6, 12, 24]:
        df[f"temp_roll_mean_{w}"] = df["temp"].shift(1).rolling(w).mean()
        df[f"temp_roll_std_{w}"] = df["temp"].shift(1).rolling(w).std()
        df[f"rhum_roll_mean_{w}"] = df["rhum"].shift(1).rolling(w).mean()
        df[f"rhum_roll_std_{w}"] = df["rhum"].shift(1).rolling(w).std()
        df[f"prcp_roll_sum_{w}"] = df["prcp"].shift(1).rolling(w).sum()
        df[f"wspd_roll_mean_{w}"] = df["wspd"].shift(1).rolling(w).mean()
        df[f"pres_roll_mean_{w}"] = df["pres"].shift(1).rolling(w).mean()

    # Loại bỏ các dòng NaN do shift/rolling tạo ra
    df = df.dropna()
    
    if df.empty:
        raise HTTPException(status_code=400, detail="Không đủ dữ liệu lịch sử để tạo features")

    # Chỉ lấy dòng cuối cùng và lọc đúng các cột FEATURES mà model yêu cầu
    return df.iloc[[-1]][FEATURES]




def internal_weather_engine(lat: float, lon: float, temps: List[float], rhums: List[float], hours: int):
    """Bộ não dự báo dùng chung cho các API"""
    history = fetch_recent_weather_history(lat, lon)
    n = len(temps)
    history.iloc[-n:, history.columns.get_loc("temp")] = temps
    history.iloc[-n:, history.columns.get_loc("rhum")] = rhums
    
    row = build_feature_row(history)
    forecast = []
    for h in range(1, hours + 1):
        forecast.append({
            "horizon": h,
            "temp": round(float(models_temp[h].predict(row)[0]), 2),
            "rhum": round(float(models_rhum[h].predict(row)[0]), 2),
            "prcp": round(max(0.0, float(models_prcp[h].predict(row)[0])), 2)
        })
    return forecast

# --- 6. API Endpoints ---

@app.get("/")
def root():
    return {"status": "online", "engine": "FastAPI + XGBoost", "provider": "Open-Meteo"}

@app.post("/smart-predict")
def smart_predict(req: SmartForecastRequest, hours: int = Query(default=6, ge=1)):
    """API nội bộ xem dự báo thời tiết thuần túy"""
    if hours > HORIZON: raise HTTPException(status_code=400, detail="Horizon limit exceeded")
    forecast = internal_weather_engine(req.lat, req.lon, req.temps, req.rhums, hours)
    return {"forecast": forecast}


@app.post("/auto-irrigation")
def auto_irrigation(req: AutoIrrigationRequest):
    # 1. Dự báo thời tiết 6h tới (Sử dụng ML Engine)
    weather_fc = internal_weather_engine(req.lat, req.lon, req.temps, req.hums, hours=6)
    
    # 2. Tính hệ số bốc hơi k_evap từ 24h lịch sử (Inference)
    diffs, drivers = [], []
    for i in range(1, len(req.moists)):
        d_m = req.moists[i-1] - req.moists[i]
        if d_m > 0:
            diffs.append(d_m)
            drivers.append(req.temps[i] / (req.hums[i] + 1))
    k_evap = sum(diffs) / sum(drivers) if drivers and sum(drivers) > 0 else 0.015
    
    # 3. GIAI ĐOẠN 1: GIẢ LẬP ĐỂ TÌM "GIỜ NGUY HIỂM"
    target_moisture = (req.min_moisture + req.max_moisture) / 2
    sim_m = req.moists[-1]
    hour_of_trouble = -1
    
    for fc in weather_fc:
        loss = k_evap * (fc["temp"] / (fc["rhum"] + 1))
        gain = fc["prcp"] * 5.0
        sim_m = sim_m - loss + gain
        if sim_m < req.min_moisture + 5:
            hour_of_trouble = fc["horizon"]
            break # Tìm thấy giờ đầu tiên gặp nguy hiểm thì dừng
            
    # 4. GIAI ĐOẠN 2: LẬP LỊCH TƯỚI SỚM (PRE-EMPTIVE)
    # Nếu có nguy hiểm, chọn giờ tưới là (giờ_nguy_hiểm - 2) hoặc giờ 1
    irrigation_hour = max(1, hour_of_trouble - 2) if hour_of_trouble > 0 else -1
    
    schedule = []
    current_m = req.moists[-1]
    
    for fc in weather_fc:
        loss = k_evap * (fc["temp"] / (fc["rhum"] + 1))
        gain = fc["prcp"] * 5.0
        
        m_before = current_m - loss + gain
        irrigation_vol = 0.0
        
        # Chỉ kích hoạt tưới tại đúng "giờ vàng" đã tính toán
        if fc["horizon"] == irrigation_hour:
            # Tưới để đưa m_before lên target
            irrigation_vol = target_moisture - m_before
            m_after = target_moisture
            status = "WATER (Pre-emptive)"
            note = f"Tưới sớm để dự phòng khô hạn vào giờ thứ {hour_of_trouble}"
        else:
            m_after = m_before
            status = "IDLE"
            note = "Duy trì theo dõi"
            
        schedule.append({
            "hour": fc["horizon"],
            "weather": fc,
            "soil_moisture_expected_before": round(m_before, 2),
            "command": status,
            "irrigation_amount": round(irrigation_vol, 2),
            "soil_moisture_expected_after": round(m_after, 2),
            "note": note
        })
        current_m = m_after # Cập nhật độ ẩm cho giờ tiếp theo trong giả lập

    # return {
    #     "analysis": {
    #         "first_danger_hour": hour_of_trouble,
    #         "scheduled_irrigation_hour": irrigation_hour,
    #         "k_evap_factor": round(k_evap, 4)
    #     },
    #     "current_status": {"moisture": req.moists[-1], "target": target_moisture},
    #     "schedule": schedule
    # }

    # Tìm thông tin tưới từ bản kế hoạch nháp
    final_status = "IDLE"
    start_hour = -1
    amount = 0.0

    if irrigation_hour != -1:
        final_status = "WATER"
        start_hour = irrigation_hour
        # Lấy lượng nước cần tưới từ đúng giờ đó trong schedule
        for item in schedule:
            if item["hour"] == irrigation_hour:
                amount = item["irrigation_amount"]
                break

    # CHỈ TRẢ VỀ CÁC THÔNG SỐ CỐT LÕI
    return {
        "status": final_status,           # "WATER" hoặc "IDLE"
        "start_at_hour": start_hour,      # Giờ bắt đầu tưới (1-6)
        "amount_pct": round(amount, 2),   # Lượng cần tưới (% độ ẩm bù thêm)
        "danger_hour": hour_of_trouble    # Giờ dự kiến sẽ bị khô (để Web hiển thị)
    }