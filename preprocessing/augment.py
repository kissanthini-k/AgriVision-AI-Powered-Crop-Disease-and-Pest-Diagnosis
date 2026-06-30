"""
Data augmentation module.

Utilizes Albumentations to create an augmentation pipeline for robust
model training, primarily handling rotations, flips, blurs, and color jitters.
"""

import os
import cv2
import albumentations as A
import glob

def get_augmentation_pipeline() -> A.Compose:
    """
    Returns an Albumentations composition pipeline.
    
    Returns:
        A.Compose object with defined augmentations.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0),
        A.CLAHE(clip_limit=2.0, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.RandomBrightnessContrast(p=0.4),
        A.Rotate(limit=20, p=0.5)
    ])

def augment_dataset(input_dir: str, output_dir: str, num_augments: int = 5) -> None:
    """
    Applies the augmentation pipeline to all images in a directory,
    saving the results to an output directory while preserving class folders.
    
    Args:
        input_dir: Path to raw dataset containing class subfolders.
        output_dir: Path to save augmented images.
        num_augments: Number of augmented copies to create per original image.
    """
    pipeline = get_augmentation_pipeline()
    
    class_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    for cls in class_folders:
        cls_in_path = os.path.join(input_dir, cls)
        cls_out_path = os.path.join(output_dir, cls)
        os.makedirs(cls_out_path, exist_ok=True)
        
        images = glob.glob(os.path.join(cls_in_path, "*.*"))
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Save original (resized)
            img_resized = cv2.resize(img_rgb, (224, 224))
            orig_out = os.path.join(cls_out_path, f"{base_name}_orig.jpg")
            cv2.imwrite(orig_out, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            
            # Generate augmentations
            for i in range(num_augments):
                augmented = pipeline(image=img_rgb)["image"]
                aug_out = os.path.join(cls_out_path, f"{base_name}_aug_{i}.jpg")
                cv2.imwrite(aug_out, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # Setup paths relative to this script
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    in_dir = os.path.join(base_dir, 'data', 'raw')
    out_dir = os.path.join(base_dir, 'data', 'augmented')
    
    print(f"Starting data augmentation...")
    print(f"Input directory: {in_dir}")
    print(f"Output directory: {out_dir}")
    print("This might take a while depending on the dataset size...")
    
    # We set num_augments=2 to prevent excessive disk usage since we have many images
    augment_dataset(in_dir, out_dir, num_augments=2)
    
    print("Augmentation completed successfully! You can now run the training script.")
