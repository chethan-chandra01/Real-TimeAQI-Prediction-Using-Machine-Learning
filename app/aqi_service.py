import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
load_dotenv()
API_KEY = os.getenv("OPEN_WEATHER_API_KEY")


def get_current_aqi(lat: float, lon: float) -> dict:
    """Fetch current air pollution data from OpenWeatherMap."""
    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={API_KEY}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def get_historical_aqi(lat: float, lon: float, days: int = 7) -> list:
    """
    Fetch historical air pollution data for the past `days` days.
    Returns a list of daily averaged component readings + AQI index.
    """
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)

    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())

    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start_ts}&end={end_ts}&appid={API_KEY}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    # Group hourly records by date and average them
    daily: dict = {}
    for entry in data.get("list", []):
        date_str = datetime.utcfromtimestamp(entry["dt"]).strftime("%Y-%m-%d")
        if date_str not in daily:
            daily[date_str] = {"count": 0, "aqi": 0, "components": {}}

        daily[date_str]["count"] += 1
        daily[date_str]["aqi"]   += entry["main"]["aqi"]
        for k, v in entry["components"].items():
            daily[date_str]["components"][k] = (
                daily[date_str]["components"].get(k, 0) + v
            )

    # Average across hours
    result = []
    for date_str in sorted(daily.keys()):
        d = daily[date_str]
        n = d["count"]
        result.append({
            "date": date_str,
            "aqi_index": round(d["aqi"] / n, 2),
            "components": {k: round(v / n, 4) for k, v in d["components"].items()}
        })

    return result


def parse_components(comp: dict) -> dict:
    """Normalize OpenWeatherMap component keys to model feature names."""
    return {
        "PM2.5": comp.get("pm2_5", 0),
        "PM10":  comp.get("pm10",  0),
        "NO":    comp.get("no",    0),
        "NO2":   comp.get("no2",   0),
        "NOx":   comp.get("no",    0) + comp.get("no2", 0),
        "NH3":   comp.get("nh3",   0),
        "CO":    comp.get("co",    0),
        "SO2":   comp.get("so2",   0),
        "O3":    comp.get("o3",    0),
    }