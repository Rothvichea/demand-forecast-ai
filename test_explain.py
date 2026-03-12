from dotenv import load_dotenv
load_dotenv()
from api.explain import analyze

# simulate a real anomaly — actual way above forecast
result = analyze(
    actual      = 384.23,   # real spike from our test data
    yhat        = 140.0,
    yhat_lower  = 62.0,
    yhat_upper  = 218.0,
    context     = {
        "timestamp":   "2018-10-22 14:00:00",
        "load_type":   "Maximum_Load",
        "hour":        14,
        "day_of_week": "Monday"
    }
)

print("Anomaly detected:", result["anomaly_detected"])
print("API called:", result["api_called"])
print("\nAnomaly info:")
for k, v in result["anomaly_info"].items():
    print(f"  {k}: {v}")

if result["explanation"]:
    print("\nClaude explanation:")
    for k, v in result["explanation"].items():
        print(f"  {k}: {v}")
