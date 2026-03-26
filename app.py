import json
import logging
import requests
import joblib
import numpy as np
import pandas as pd
import os

from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- 1. Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("irrigation_pro")

ARTIFACT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ARTIFACT_DIR / "models"
DB_FILE = ARTIFACT_DIR / "irrigation_db.json"

if not DB_FILE.exists():
    with open(DB_FILE, "w") as f: json.dump({}, f)

with open(ARTIFACT_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
FEATURES = metadata["features"]
models_temp, models_rhum, models_prcp = {}, {}, {}

for h in range(1, metadata["horizon"] + 1):
    models_temp[h] = joblib.load(MODEL_DIR / f"model_temp_h{h}.joblib")
    models_rhum[h] = joblib.load(MODEL_DIR / f"model_rhum_h{h}.joblib")
    models_prcp[h] = joblib.load(MODEL_DIR / f"model_prcp_h{h}.joblib")

app = FastAPI(title="Pro Irrigation AI v3.5")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 2. Request Models ---
class AutoIrrigationRequest(BaseModel):
    zone: str
    lat: float; lon: float
    min_moisture: float; max_moisture: float
    temps: List[float]; hums: List[float]; moists: List[float]

class SmartForecastRequest(BaseModel):
    lat: float; lon: float; temps: List[float]; rhums: List[float]

# --- 3. Database Helpers ---
def clear_zone(zone: str):
    with open(DB_FILE, "r") as f: db = json.load(f)
    if zone in db:
        del db[zone]
        with open(DB_FILE, "w") as f: json.dump(db, f, indent=2)

def save_schedule(zone: str, scheduled_at: datetime, amount: float, danger_h: int):
    with open(DB_FILE, "r") as f: db = json.load(f)
    db[zone] = {
        "status": "WATER",
        "scheduled_at": scheduled_at.isoformat(),
        "expires_at": (scheduled_at + timedelta(minutes=45)).isoformat(),
        "amount_pct": round(amount, 2),
        "danger_hour": danger_h,
        "created_at": datetime.now().isoformat()
    }
    with open(DB_FILE, "w") as f: json.dump(db, f, indent=2)

# --- 4. Core Logic Functions ---

def calc_k_evap(temps_hist, hums_hist, moists_hist):
    diffs, drivers = [], []
    for i in range(1, len(moists_hist)):
        d_m = moists_hist[i-1] - moists_hist[i]
        if d_m > 0:
            diffs.append(d_m)
            drivers.append(temps_hist[i] / (hums_hist[i] + 1))
    return sum(diffs) / sum(drivers) if drivers and sum(drivers) > 0 else 0.015

def fetch_recent_weather_history(lat, lon, lookback_hours=36):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m", "past_days": 2, "forecast_days": 1, "timezone": "UTC"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status() # Kiểm tra lỗi HTTP
        data = resp.json()["hourly"]
        df = pd.DataFrame({"temp": data["temperature_2m"], "rhum": data["relative_humidity_2m"], "prcp": data["precipitation"], "pres": data["surface_pressure"], "wspd": data["wind_speed_10m"]}, index=pd.to_datetime(data["time"])).astype(float)
        return df.sort_index().interpolate().ffill().bfill().tail(lookback_hours + 12)
    except Exception as e:
        logger.error(f"Weather API Error: {e}")
        raise HTTPException(status_code=502, detail="Weather provider unavailable")

def build_feature_row(df):
    df = df.copy()
    df["hour"], df["day_of_week"], df["month"], df["day_of_year"] = df.index.hour, df.index.dayofweek, df.index.month, df.index.dayofyear
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * df["hour"] / 24), np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * df["day_of_year"] / 366), np.cos(2 * np.pi * df["day_of_year"] / 366)
    
    use_cols = ["temp", "rhum", "prcp", "wspd", "pres"]
    for lag in [1, 2, 3, 6, 12, 24]:
        for col in use_cols: df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    
    # --- BỔ SUNG ĐẦY ĐỦ ROLLING WINDOWS ĐỂ TRÁNH KEYERROR ---
    for w in [3, 6, 12, 24]:
        df[f"temp_roll_mean_{w}"] = df["temp"].shift(1).rolling(w).mean()
        df[f"temp_roll_std_{w}"] = df["temp"].shift(1).rolling(w).std()
        df[f"rhum_roll_mean_{w}"] = df["rhum"].shift(1).rolling(w).mean()
        df[f"rhum_roll_std_{w}"] = df["rhum"].shift(1).rolling(w).std()
        df[f"prcp_roll_sum_{w}"] = df["prcp"].shift(1).rolling(w).sum()
        df[f"wspd_roll_mean_{w}"] = df["wspd"].shift(1).rolling(w).mean()
        df[f"pres_roll_mean_{w}"] = df["pres"].shift(1).rolling(w).mean()

    return df.dropna().iloc[[-1]][FEATURES]

def internal_weather_engine(lat, lon, temps, rhums, hours):
    history = fetch_recent_weather_history(lat, lon)
    n = len(temps)
    history.iloc[-n:, history.columns.get_loc("temp")] = temps
    history.iloc[-n:, history.columns.get_loc("rhum")] = rhums
    row = build_feature_row(history)
    return [{"horizon": h, "temp": round(float(models_temp[h].predict(row)[0]), 2), "rhum": round(float(models_rhum[h].predict(row)[0]), 2), "prcp": round(max(0.0, float(models_prcp[h].predict(row)[0])), 2)} for h in range(1, hours + 1)]

# --- 5. Endpoints ---

@app.get("/")
def root():
    return {"status": "online", "message": "Pro Irrigation AI v3.5 is running"}

@app.post("/smart-predict")
def smart_predict(req: SmartForecastRequest, hours: int = Query(default=6, ge=1)):
    forecast = internal_weather_engine(req.lat, req.lon, req.temps, req.rhums, hours)
    return {"forecast": forecast}

@app.post("/auto-irrigation")
def auto_irrigation(req: AutoIrrigationRequest):
    now = datetime.now()
    target_moisture = (req.min_moisture + req.max_moisture) / 2
    
    # BƯỚC 0: KIỂM TRA PENDING
    with open(DB_FILE, "r") as f: db = json.load(f)
    if req.zone in db:
        pending = db[req.zone]
        sched_time = datetime.fromisoformat(pending["scheduled_at"])
        expire_time = datetime.fromisoformat(pending["expires_at"])
        
        is_expired = now > expire_time
        is_reached_target = req.moists[-1] >= target_moisture - 2
        
        if not is_expired and not is_reached_target:
            rem_min = int((sched_time - now).total_seconds() / 60)
            return {"status": "WATER", "start_in_minutes": max(0, rem_min), "amount_pct": pending["amount_pct"], "danger_hour": pending["danger_hour"], "source": "verified_pending"}
        else:
            clear_zone(req.zone)

    # BƯỚC 1: DỰ BÁO VÀ LẬP LỊCH
    forecast = internal_weather_engine(req.lat, req.lon, req.temps, req.hums, 6)
    k_evap = calc_k_evap(req.temps, req.hums, req.moists)
    
    # Tìm danger_hour
    sim_m, hour_of_trouble = req.moists[-1], -1
    for fc in forecast:
        sim_m = sim_m - (k_evap * (fc["temp"] / (fc["rhum"] + 1))) + (fc["prcp"] * 5.0)
        if sim_m < req.min_moisture + 5:
            hour_of_trouble = fc["horizon"]; break
    
    if hour_of_trouble != -1:
        lead_h = max(1, hour_of_trouble - 2)
        sched_at = now + timedelta(hours=lead_h)
        
        # Ước tính amount
        m_at_sched = req.moists[-1]
        for i in range(lead_h):
            fc = forecast[i]
            m_at_sched = m_at_sched - (k_evap * (fc["temp"] / (fc["rhum"] + 1))) + (fc["prcp"] * 5.0)
        
        amount = max(0, target_moisture - m_at_sched)
        save_schedule(req.zone, sched_at, amount, hour_of_trouble)
        
        return {"status": "WATER", "start_in_minutes": lead_h * 60, "amount_pct": round(amount, 2), "danger_hour": hour_of_trouble, "scheduled_at_abs": sched_at.isoformat()}
    
    return {"status": "IDLE", "start_in_minutes": 0, "amount_pct": 0, "danger_hour": -1}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)