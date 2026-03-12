from dotenv import load_dotenv
load_dotenv()  # ← loads .env before anything else

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.forecast import router as forecast_router

app = FastAPI(
    title="Demand Forecast AI",
    description="Forecasting API with LLM-powered anomaly explanation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast_router, prefix="/forecast", tags=["forecast"])

@app.get("/health")
def health():
    return {"status": "ok"}