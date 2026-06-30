"""
Damage assessment module.

Maps model prediction confidence to severity levels and extracts corresponding
yield loss estimations from the local database.
"""

def assess(disease_key: str, confidence: float, db: dict) -> dict:
    """
    Determines severity level based on prediction confidence and retrieves
    associated loss metrics.
    
    Args:
        disease_key: The string identifier for the disease/pest.
        confidence: Prediction confidence score [0.0 - 1.0].
        db: The loaded disease database dictionary.
        
    Returns:
        Dictionary containing severity, yield_loss_pct, inr_per_acre, and description.
    """
    if disease_key not in db:
        return {"severity_stage": "unknown", "severity": "unknown", "yield_loss_pct": "0", "inr_per_acre": "0", "description": "No data"}
        
    entry = db[disease_key]
    
    if entry.get("type") == "pest":
        # Severity mapped roughly to confidence as a proxy for visual extent
        if confidence > 0.85:
            stage = "late"
        elif confidence > 0.70:
            stage = "mid"
        else:
            stage = "early"
    else:
        if confidence > 0.85:
            stage = "late"
        elif confidence > 0.70:
            stage = "mid"
        else:
            stage = "early"
            
    severity_data = entry["severity"][stage]
    severity_data["severity_stage"] = stage
    
    return severity_data
