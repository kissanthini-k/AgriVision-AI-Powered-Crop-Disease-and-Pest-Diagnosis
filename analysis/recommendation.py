"""
Recommendation module.

Fetches disease treatments (chemical and organic) and translated names
based on the requested regional language.
"""

def get_recommendation(disease_key: str, language: str, db: dict) -> dict:
    """
    Retrieves translated name and solution dictionary for a specific language.
    
    Args:
        disease_key: ID of the disease.
        language: Two-letter language code (e.g., 'hi', 'ta').
        db: Loaded disease database.
        
    Returns:
        Dictionary with English name, translated name, and solution steps.
    """
    if disease_key not in db:
        return {
            "name_en": "Unknown",
            "name_regional": "Unknown",
            "solution": {}
        }
        
    entry = db[disease_key]
    
    name_en = entry.get("en", "Unknown")
    name_regional = entry.get(language, name_en) # Fallback to English
    
    solution = entry.get("solution", {})
    
    return {
        "name_en": name_en,
        "name_regional": name_regional,
        "solution": solution,
        "type": entry.get("type", "unknown")
    }
