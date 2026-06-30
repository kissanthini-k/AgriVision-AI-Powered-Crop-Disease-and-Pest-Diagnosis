import os
import shutil
import glob

DATA_RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))

def flatten_dataset():
    print(f"Scanning {DATA_RAW_DIR} for nested image folders...")
    
    # Find all directories that contain images
    all_images = glob.glob(os.path.join(DATA_RAW_DIR, '**', '*.jpg'), recursive=True) + \
                 glob.glob(os.path.join(DATA_RAW_DIR, '**', '*.png'), recursive=True) + \
                 glob.glob(os.path.join(DATA_RAW_DIR, '**', '*.jpeg'), recursive=True)
                 
    leaf_dirs = set(os.path.dirname(img_path) for img_path in all_images)
    print(f"Found {len(leaf_dirs)} leaf directories containing images.")
    
    moved_count = 0
    collision_count = 0
    
    for leaf in leaf_dirs:
        # If it's already directly inside DATA_RAW_DIR, skip it
        if os.path.dirname(leaf) == DATA_RAW_DIR:
            continue
            
        class_name = os.path.basename(leaf)
        # Avoid naming collisions by prepending parent if needed, but let's try direct first
        target_dir = os.path.join(DATA_RAW_DIR, class_name)
        
        if os.path.exists(target_dir) and target_dir != leaf:
            # Collision! Let's prefix with parent folder name to make it unique
            parent_name = os.path.basename(os.path.dirname(leaf))
            unique_name = f"{parent_name}_{class_name}"
            target_dir = os.path.join(DATA_RAW_DIR, unique_name)
            collision_count += 1
            
        print(f"Moving: {os.path.relpath(leaf, DATA_RAW_DIR)} -> {os.path.basename(target_dir)}")
        try:
            shutil.move(leaf, target_dir)
            moved_count += 1
        except Exception as e:
            print(f"Failed to move {leaf}: {e}")
            
    print(f"Flattening complete! Moved {moved_count} folders to the root of data/raw/. Resolved {collision_count} collisions.")
    
    # Cleanup empty directories
    print("Cleaning up empty nested directories...")
    for root, dirs, files in os.walk(DATA_RAW_DIR, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):
                try:
                    os.rmdir(dir_path)
                except OSError:
                    pass

if __name__ == "__main__":
    flatten_dataset()
