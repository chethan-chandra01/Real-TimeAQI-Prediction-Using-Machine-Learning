from fastapi import FastAPI, HTTPException
from .aqi_service import get_current_aqi, get_historical_aqi, parse_components
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

app = FastAPI(
    title="AQI Forecast API",
    description="Predicts tomorrow's AQI using today's and historical pollutant data.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/ui")
def frontend():
    return FileResponse("index.html")
model    = joblib.load("aqi_forecast_xgb.pkl")
FEATURES = joblib.load("aqi_forecast_features.pkl")

def aqi_bucket(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Satisfactory"
    if aqi <= 200:  return "Moderate"
    if aqi <= 300:  return "Poor"
    if aqi <= 400:  return "Very Poor"
    return "Severe"

@app.get("/")
def home():
    return {"message": "AQI Forecast API v2 — use /predict_aqi or /forecast_tomorrow"}


@app.get("/predict_aqi")
def predict_aqi_current(lat: float, lon: float):
    """Predict AQI from current pollutant readings (original endpoint)."""
    try:
        data      = get_current_aqi(lat, lon)
        pollution = data["list"][0]
        comp      = pollution["components"]
        now       = datetime.now()

        input_data = pd.DataFrame([{
            **parse_components(comp),
            "year":  now.year,
            "month": now.month,
            "day":   now.day
        }])

        # Keep only the columns the model expects
        for col in FEATURES:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[FEATURES]

        predicted_aqi = float(model.predict(input_data)[0])
        return {
            "predicted_aqi": round(predicted_aqi, 1),
            "aqi_category":  aqi_bucket(predicted_aqi)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast_tomorrow")
def forecast_tomorrow(lat: float, lon: float):
    """
    Predict tomorrow's AQI using today's readings + last 7 days of history.
    This is the main forecasting endpoint.
    """
    try:
        # ── Fetch data ──────────────────────────────────────────────────────
        current_data = get_current_aqi(lat, lon)
        history      = get_historical_aqi(lat, lon, days=7)

        if len(history) < 2:
            raise HTTPException(
                status_code=400,
                detail="Not enough historical data. Need at least 2 days."
            )

        # ── Build AQI history list (most recent last) ───────────────────────
        # OpenWeatherMap AQI index: 1=Good … 5=Very Poor → scale to ~0–500
        ow_to_india = {1: 30, 2: 75, 3: 150, 4: 250, 5: 350}
        aqi_history = [
            ow_to_india.get(int(round(d["aqi_index"])), 150)
            for d in history
        ]

        today_aqi = aqi_history[-1]

        # ── Today's pollutants ──────────────────────────────────────────────
        today_comp = parse_components(
            current_data["list"][0]["components"]
        )

        # ── Compute lag & rolling features ─────────────────────────────────
        now = datetime.now()

        def safe_get(lst, idx):
            """Get element from end of list by negative index, or 0."""
            try:
                return lst[idx] if abs(idx) <= len(lst) else lst[0]
            except IndexError:
                return 0

        aqi_lag1 = safe_get(aqi_history, -1)
        aqi_lag2 = safe_get(aqi_history, -2)
        aqi_lag3 = safe_get(aqi_history, -3)
        aqi_lag7 = safe_get(aqi_history, -7)

        roll3  = float(np.mean(aqi_history[-3:]))
        roll7  = float(np.mean(aqi_history[-7:]))
        roll14 = roll7   # only 7 days available; use same
        std7   = float(np.std(aqi_history[-7:]))  if len(aqi_history) >= 2 else 0

        # Yesterday's pollutants (last full day in history)
        yesterday_comp = parse_components(history[-1]["components"])

        row = {
            # Current pollutants
            **today_comp,
            # AQI lags
            "AQI_lag1": aqi_lag1,
            "AQI_lag2": aqi_lag2,
            "AQI_lag3": aqi_lag3,
            "AQI_lag7": aqi_lag7,
            # Rolling stats
            "AQI_roll3":  roll3,
            "AQI_roll7":  roll7,
            "AQI_roll14": roll14,
            "AQI_std7":   std7,
            # Pollutant lags (yesterday)
            "PM2.5_lag1": yesterday_comp["PM2.5"],
            "CO_lag1":    yesterday_comp["CO"],
            "PM10_lag1":  yesterday_comp["PM10"],
            "NO2_lag1":   yesterday_comp["NO2"],
            "O3_lag1":    yesterday_comp["O3"],
            # Temporal
            "month":       now.month,
            "day_of_week": now.weekday(),
            "day_of_year": now.timetuple().tm_yday,
            "season": {
                12: 0, 1: 0, 2: 0,
                3: 1, 4: 1, 5: 1,
                6: 2, 7: 2, 8: 2, 9: 2,
                10: 3, 11: 3
            }[now.month]
        }

        input_df = pd.DataFrame([row])

        for col in FEATURES:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[FEATURES]

        predicted_aqi = float(model.predict(input_df)[0])
        predicted_aqi = max(0, predicted_aqi)   # AQI can't be negative

        return {
            "forecast_date":       (datetime.now()+ timedelta(days=1)).strftime("%Y-%m-%d"),
            "predicting_for":      "tomorrow",
            "today_aqi_estimate":  today_aqi,
            "predicted_aqi":       round(predicted_aqi, 1),
            "aqi_category":        aqi_bucket(predicted_aqi),
            "trend":               (
                "Improving" if predicted_aqi < today_aqi - 10
                else "Worsening" if predicted_aqi > today_aqi + 10
                else "Stable"
            ),
            "key_inputs": {
                "today_pm25": today_comp["PM2.5"],
                "today_co":   today_comp["CO"],
                "7day_avg":   round(roll7, 1),
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))