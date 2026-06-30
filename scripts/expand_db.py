import os
import json
import glob
import re

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'disease_db.json'))

def sanitize_name(name):
    # Remove any special characters and replace spaces with underscores
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', name)
    clean = clean.strip().replace(' ', '_').lower()
    return clean

def generate_generic_entry(original_name, is_pest=False):
    # Create a nice English name
    english_name = original_name.replace('_', ' ').title()
    
    return {
        "en": english_name,
        "hi": f"सामान्य (Generic): {english_name}",
        "ta": f"பொதுவான (Generic): {english_name}",
        "te": f"సాధారణ (Generic): {english_name}",
        "kn": f"ಸಾಮಾನ್ಯ (Generic): {english_name}",
        "mr": f"सामान्य (Generic): {english_name}",
        "type": "pest" if is_pest else "disease",
        "crops": ["unknown"],
        "season": ["kharif", "rabi"],
        "agro_zones": ["All Regions"],
        "severity": {
            "early": {"yield_loss_pct": 10, "inr_per_acre": 2000, "description": "Early symptoms observed. Monitor closely."},
            "mid": {"yield_loss_pct": 30, "inr_per_acre": 6000, "description": "Moderate damage visible on plant structures."},
            "late": {"yield_loss_pct": 60, "inr_per_acre": 15000, "description": "Severe infestation/infection causing massive damage."}
        },
        "solution": {
            "pesticide": "Broad-spectrum control agent or consult local agricultural officer.",
            "application": "Apply as per generic guidelines for this type of issue.",
            "organic_alternative": "Neem oil spray (1500 ppm) @ 5ml/L or Trichoderma soil application.",
            "source": "Auto-Generated DB"
        }
    }

def main():
    print("--- Scanning for image directories ---")
    
    # We will use os.walk to find all directories that contain images
    directories_with_images = set()
    for root, dirs, files in os.walk(DATASET_ROOT):
        # Check if there's any image in this directory
        has_image = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)
        if has_image:
            directories_with_images.add(root)
            
    print(f"Found {len(directories_with_images)} raw class directories containing images.")
    
    # Load existing DB
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r', encoding='utf-8') as f:
            db = json.load(f)
    else:
        db = {}
        
    added_count = 0
    
    for dir_path in directories_with_images:
        folder_name = os.path.basename(dir_path)
        
        # Skip weird train/test split folders like 'train_images' unless they are the actual class
        # Wait, if the folder is literally 'train_images' or 'val_images', the actual class name is likely the parent
        if folder_name.lower() in ['train', 'test', 'val', 'train_images', 'val_images', 'images', 'train_set']:
            # The class name is the parent folder
            class_name_raw = os.path.basename(os.path.dirname(dir_path))
        else:
            class_name_raw = folder_name
            
        sanitized_id = sanitize_name(class_name_raw)
        
        # Skip empty strings
        if not sanitized_id:
            continue
            
        # Check if it's likely a pest based on path
        is_pest = 'pest' in dir_path.lower()
        
        if sanitized_id not in db:
            print(f"Adding new auto-generated class: {sanitized_id}")
            db[sanitized_id] = generate_generic_entry(class_name_raw, is_pest)
            added_count += 1
            
    # Save the expanded DB
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
        
    print(f"Done! Added {added_count} new entries to disease_db.json.")

if __name__ == "__main__":
    main()
