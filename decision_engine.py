from typing import List, Dict, Any


def summarize_forecast(forecast: List[Dict[str, Any]]) -> Dict[str, float]:
    max_temp = max(x["pred_temp"] for x in forecast)
    min_temp = min(x["pred_temp"] for x in forecast)
    max_rhum = max(x["pred_rhum"] for x in forecast)
    max_prcp = max(x["pred_prcp"] for x in forecast)
    total_prcp = sum(x["pred_prcp"] for x in forecast)

    rainy_hours = sum(1 for x in forecast if x["pred_prcp"] >= 0.1)
    hot_hours = sum(1 for x in forecast if x["pred_temp"] >= 35)
    humid_hours = sum(1 for x in forecast if x["pred_rhum"] >= 85)

    return {
        "max_temp": max_temp,
        "min_temp": min_temp,
        "max_rhum": max_rhum,
        "max_prcp": max_prcp,
        "total_prcp": total_prcp,
        "rainy_hours": rainy_hours,
        "hot_hours": hot_hours,
        "humid_hours": humid_hours,
    }


def apply_rules(forecast: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = summarize_forecast(forecast)

    decisions = []
    reasons = []
    severity = "low"

    # Rule 1: mưa lớn
    if stats["max_prcp"] >= 10 or stats["total_prcp"] >= 20:
        decisions.append("RAIN_ALERT")
        reasons.append("Forecast indicates heavy rainfall in the next hours.")
        severity = "high"

    # Rule 2: mưa vừa / có khả năng ảnh hưởng
    elif stats["rainy_hours"] >= 3 or stats["max_prcp"] >= 3:
        decisions.append("BRING_UMBRELLA")
        reasons.append("Forecast indicates likely rain within the requested horizon.")
        if severity != "high":
            severity = "medium"

    # Rule 3: nóng
    if stats["max_temp"] >= 35:
        decisions.append("HEAT_ALERT")
        reasons.append("Forecast temperature may reach a high heat threshold.")
        severity = "high"

    # Rule 4: nóng ẩm
    if stats["max_temp"] >= 33 and stats["max_rhum"] >= 75:
        decisions.append("LIMIT_OUTDOOR_ACTIVITY")
        reasons.append("Hot and humid conditions may reduce comfort outdoors.")
        if severity == "low":
            severity = "medium"

    # Rule 5: rất ẩm
    if stats["max_rhum"] >= 90:
        decisions.append("HIGH_HUMIDITY_NOTICE")
        reasons.append("Humidity is expected to remain very high.")
        if severity == "low":
            severity = "medium"

    if not decisions:
        decisions.append("NORMAL")
        reasons.append("No strong weather risk detected from current rules.")

    return {
        "summary": stats,
        "decision": decisions,
        "severity": severity,
        "reasons": reasons,
    }