"""
Confidence gating module.

Validates if the model is confident enough to proceed with a diagnosis.
Prevents the system from confidently making incorrect diagnoses on edge cases.
"""

def check_confidence(confidence: float, threshold: float = 0.10) -> dict:
    """
    Checks if the prediction confidence meets the minimum threshold.
    
    Args:
        confidence: The confidence score [0.0 - 1.0].
        threshold: The minimum acceptable confidence.
        
    Returns:
        A dictionary with the gate status.
    """
    if confidence >= threshold:
        return {
            "status": "confident", 
            "proceed": True
        }
    else:
        return {
            "status": "uncertain", 
            "proceed": False, 
            "message": "upload a valid image"
        }
