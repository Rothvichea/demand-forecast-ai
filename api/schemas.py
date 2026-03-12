from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class ForecastPoint(BaseModel):
    ds: datetime
    yhat: float
    yhat_lower: Optional[float] = None
    yhat_upper: Optional[float] = None
    anomaly: bool = False


class AnomalyExplanation(BaseModel):
    anomaly_detected: bool
    severity: str          # low | medium | high
    explanation: str
    recommended_action: str


class ForecastResponse(BaseModel):
    model: str             # "prophet" | "lstm"
    n_points: int
    mae: Optional[float] = None
    forecast: List[ForecastPoint]
    explanation: Optional[AnomalyExplanation] = None
