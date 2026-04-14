@echo off
echo Starting TradeIQ FastAPI Backend...
start /B uvicorn api.app:app --reload --port 8000

echo Waiting for backend to initialize...
timeout /t 3 /nobreak > NUL

echo Starting TradeIQ Streamlit Frontend...
streamlit run app/streamlit_app.py

echo TradeIQ is running! Close this window to stop.
pause
