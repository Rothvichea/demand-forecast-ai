import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def detect_anomaly(actual: float, yhat: float,
                   yhat_lower: float, yhat_upper: float) -> dict:
    """
    Check if actual demand falls outside Prophet 95% CI
    Returns anomaly info dict
    """
    is_anomaly = (actual < yhat_lower) or (actual > yhat_upper)
    deviation  = actual - yhat
    pct_dev    = abs(deviation) / yhat * 100 if yhat > 0 else 0

    return {
        "is_anomaly":   is_anomaly,
        "actual":       round(actual, 2),
        "forecast":     round(yhat, 2),
        "lower_bound":  round(yhat_lower, 2),
        "upper_bound":  round(yhat_upper, 2),
        "deviation_kwh": round(deviation, 2),
        "deviation_pct": round(pct_dev, 1)
    }


def explain_anomaly(anomaly_info: dict, context: dict) -> dict:
    """
    Call Claude API only when anomaly is detected.
    Returns structured JSON explanation.
    """
    prompt = f"""You are an industrial energy analyst for a steel factory.

An anomaly has been detected in the factory's energy consumption.

Anomaly details:
- Timestamp: {context.get('timestamp')}
- Actual demand: {anomaly_info['actual']} kWh
- Forecast demand: {anomaly_info['forecast']} kWh
- Expected range: {anomaly_info['lower_bound']} — {anomaly_info['upper_bound']} kWh
- Deviation: {anomaly_info['deviation_kwh']} kWh ({anomaly_info['deviation_pct']}% off forecast)
- Load type: {context.get('load_type', 'unknown')}
- Hour of day: {context.get('hour', 'unknown')}
- Day of week: {context.get('day_of_week', 'unknown')}

Return ONLY a valid JSON object with exactly these fields:
{{
  "severity": "low" or "medium" or "high",
  "anomaly_type": "overconsumption" or "underconsumption",
  "explanation": "2-3 sentence plain language explanation for a factory manager",
  "likely_cause": "most probable cause in one sentence",
  "recommended_action": "concrete action the manager should take"
}}

Rules:
- severity low: deviation < 20%
- severity medium: deviation 20-50%
- severity high: deviation > 50%
- Be specific, practical and concise
- No markdown, no preamble, just the JSON"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # fallback if Claude adds extra text
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(match.group()) if match else {"error": raw}

    return result


def analyze(actual: float, yhat: float, yhat_lower: float,
            yhat_upper: float, context: dict) -> dict:
    """
    Main entry point — called from the API endpoint.
    Only calls Claude if anomaly is detected.
    """
    anomaly = detect_anomaly(actual, yhat, yhat_lower, yhat_upper)

    if not anomaly["is_anomaly"]:
        return {
            "anomaly_detected": False,
            "anomaly_info":     anomaly,
            "explanation":      None,
            "api_called":       False
        }

    # anomaly detected — call Claude
    explanation = explain_anomaly(anomaly, context)

    return {
        "anomaly_detected": True,
        "anomaly_info":     anomaly,
        "explanation":      explanation,
        "api_called":       True
    }