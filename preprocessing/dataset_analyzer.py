"""
High-Performance Dataset Analyzer.

Scans the dataset directory to detect and optionally remove:
- Corrupted images (cannot be opened by PIL)
- Invalid extensions
- Exact duplicates (via MD5 hashing)
- Empty class folders
"""

import os
import hashlib
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm
import time

def process_file(filepath: str, delete_invalid: bool = False):
    """Processes a single file to check for validity and compute MD5."""
    result = {
        "filepath": filepath,
        "valid": True,
        "md5": None,
        "error": None,
        "deleted": False
    }
    
    # Check extension
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in valid_exts:
        result["valid"] = False
        result["error"] = "Invalid extension"
    else:
        # Check corruption
        try:
            with Image.open(filepath) as img:
                img.verify()
        except Exception as e:
            result["valid"] = False
            result["error"] = f"Corrupt image: {str(e)}"
            
    # Compute MD5 if valid
    if result["valid"]:
        try:
            hasher = hashlib.md5()
            with open(filepath, 'rb') as f:
                buf = f.read(65536)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(65536)
            result["md5"] = hasher.hexdigest()
        except Exception as e:
            result["valid"] = False
            result["error"] = f"Failed to hash: {str(e)}"
            
    # Delete if invalid and requested
    if not result["valid"] and delete_invalid:
        try:
            os.remove(filepath)
            result["deleted"] = True
        except Exception as e:
            pass
            
    return result

def analyze_dataset(data_dir: str, delete_invalid: bool = False, delete_duplicates: bool = False, num_workers: int = 8):
    print(f"Starting dataset analysis on {data_dir}...")
    start_time = time.time()
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} does not exist.")
        return
        
    all_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
            
    total_files = len(all_files)
    print(f"Found {total_files} files to analyze.")
    
    stats = {
        "total_files": total_files,
        "valid_images": 0,
        "corrupted_or_invalid": 0,
        "duplicates_found": 0,
        "files_deleted": 0
    }
    
    hash_map = {}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, f, delete_invalid): f for f in all_files}
        
        for future in tqdm(as_completed(futures), total=total_files, desc="Analyzing images"):
            res = future.result()
            
            if not res["valid"]:
                stats["corrupted_or_invalid"] += 1
                if res["deleted"]:
                    stats["files_deleted"] += 1
            else:
                md5 = res["md5"]
                if md5 in hash_map:
                    stats["duplicates_found"] += 1
                    if delete_duplicates:
                        try:
                            os.remove(res["filepath"])
                            stats["files_deleted"] += 1
                        except:
                            pass
                else:
                    hash_map[md5] = res["filepath"]
                    stats["valid_images"] += 1
                    
    # Check for empty folders
    empty_folders = 0
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):
                empty_folders += 1
                if delete_invalid:
                    try:
                        os.rmdir(dir_path)
                    except:
                        pass
    stats["empty_folders_found"] = empty_folders
    
    duration = time.time() - start_time
    stats["analysis_time_seconds"] = round(duration, 2)
    
    print("\n--- Analysis Complete ---")
    print(json.dumps(stats, indent=4))
    
    with open("dataset_health.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print("Saved health report to dataset_health.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/augmented", help="Dataset directory to analyze")
    parser.add_argument("--delete_invalid", action="store_true", help="Delete corrupted files and empty folders")
    parser.add_argument("--delete_duplicates", action="store_true", help="Delete exact duplicate files")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of parallel workers")
    args = parser.parse_args()
    
    analyze_dataset(args.data_dir, args.delete_invalid, args.delete_duplicates, args.workers)
