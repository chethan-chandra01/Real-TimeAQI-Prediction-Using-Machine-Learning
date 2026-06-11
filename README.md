# Real-TimeAQI-Prediction-Using-Machine-Learning
A machine learning system that predicts tomorrow's Air Quality Index (AQI) for major Indian cities using live pollutant data and historical patterns. Built with XGBoost, FastAPI, and a Vanilla JS frontend with an interactive city selection interface.
Overview
This project addresses a practical gap in India's air quality monitoring ecosystem — while real-time AQI data is widely available, next-day forecasts are not. The system fetches live pollution readings from the OpenWeatherMap API, constructs time-series lag features from the past 7 days of historical data, and passes them through a trained XGBoost model to predict tomorrow's AQI along with its CPCB health category and trend direction

Features:

Next-day AQI forecasting for 30+ major Indian cities
Real-time data integration via OpenWeatherMap Air Pollution API
XGBoost model trained on 29,531 CPCB records (2015–2020)
Time-series feature engineering: lag features, rolling averages, seasonal encodings
FastAPI backend with auto-generated Swagger documentation at /docs
Interactive city selection sidebar with state-based grouping and live search
Color-coded AQI result dashboard with trend indicator and pollutant inputs
Self-contained single-file HTML frontend — no build tools required

