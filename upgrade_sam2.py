"""
SAM2 Model Upgrade Script
Download and configure larger SAM2 model for better segmentation
"""

import os
from pathlib import Path
import requests
from tqdm import tqdm

def download_sam2_base():
    """Download SAM2 Hiera Base checkpoint (larger, more accurate)"""
    print("="*70)
    print("SAM2 Model Upgrade")
    print("="*70)
    
    checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    checkpoint_path = "sam2.1_hiera_base_plus.pt"
    
    if Path(checkpoint_path).exists():
        print(f"✅ {checkpoint_path} already exists ({Path(checkpoint_path).stat().st_size / 1e6:.1f} MB)")
        return checkpoint_path
    
    print(f"\n📥 Downloading SAM2 Hiera Base Plus...")
    print(f"URL: {checkpoint_url}")
    print(f"Target: {checkpoint_path}")
    print("\nThis is a larger model (~321 MB) with better accuracy than Tiny")
    
    response = requests.get(checkpoint_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(checkpoint_path, 'wb') as f, tqdm(
        desc=checkpoint_path,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)
    
    print(f"\n✅ Downloaded {checkpoint_path}")
    return checkpoint_path

def update_sam2_config():
    """Update sam2_segmentation.py to use the larger model"""
    print("\n📝 Updating SAM2 configuration...")
    
    # Backup original
    original_file = "sam2_segmentation.py"
    if Path(original_file).exists():
        backup_file = "sam2_segmentation_backup.py"
        if not Path(backup_file).exists():
            Path(backup_file).write_text(Path(original_file).read_text())
            print(f"✅ Backed up to {backup_file}")
    
    # Create updated version
    updated_code = '''import torch
import numpy as np
import cv2

class SAM2Segmenter:
    """
    Production-quality segmentation using Meta's SAM2 (Base Plus model).
    """
    
    def __init__(self, checkpoint_path="sam2.1_hiera_base_plus.pt", device="cpu"):
        """Initialize SAM2 model with larger checkpoint"""
        self.device = device
        
        print(f"Loading SAM2 Base Plus model on {device}...")
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # Use larger model
        self.predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-base-plus",
            device=device
        )
        
        print("SAM2 Base Plus initialized successfully!")
    
    def segment(self, image_path):
        """
        Generate masks with optimized parameters for blueprints
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image
        self.predictor.set_image(image_rgb)
        
        # Optimized parameters for architectural blueprints
        h, w = image_rgb.shape[:2]
        points_per_side = 20  # Increased density
        step = max(h, w) // points_per_side
        
        masks_list = []
        seen_regions = set()
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                try:
                    masks, scores, _ = self.predictor.predict(
                        point_coords=np.array([[x, y]]),
                        point_labels=np.array([1]),
                        multimask_output=True  # Get multiple mask proposals
                    )
                    
                    # Take best mask
                    if len(masks) > 0:
                        best_idx = np.argmax(scores)
                        mask = masks[best_idx]
                        score = scores[best_idx]
                        
                        if score > 0.7:  # Higher confidence threshold
                            # Check for duplicate regions
                            mask_hash = hash(mask.tobytes())
                            if mask_hash not in seen_regions:
                                seen_regions.add(mask_hash)
                                
                                y_indices, x_indices = np.where(mask)
                                if len(y_indices) > 100:  # Минимум пикселей
                                    bbox = [
                                        int(x_indices.min()),
                                        int(y_indices.min()),
                                        int(x_indices.max() - x_indices.min()),
                                        int(y_indices.max() - y_indices.min())
                                    ]
                                    masks_list.append({
                                        'segmentation': mask,
                                        'area': int(mask.sum()),
                                        'bbox': bbox,
                                        'predicted_iou': float(score)
                                    })
                except:
                    continue
        
        # Sort by area and keep largest masks
        masks_list.sort(key=lambda x: x['area'], reverse=True)
        masks_list = masks_list[:100]  # Top 100 masks
        
        print(f"Generated {len(masks_list)} high-quality masks")
        return masks_list
'''
    
    with open("sam2_segmentation_upgraded.py", 'w') as f:
        f.write(updated_code)
    
    print(f"✅ Created sam2_segmentation_upgraded.py")
    print("\nℹ️  To use the upgraded model, update app.py to import from sam2_segmentation_upgraded")

if __name__ == "__main__":
    # Download larger model
    checkpoint = download_sam2_base()
    
    # Update configuration
    update_sam2_config()
    
    print("\n" + "="*70)
    print("✅ SAM2 Upgrade Complete!")
    print("="*70)
    print("\nBenefits of Base Plus model:")
    print("  • Better edge detection for room boundaries")
    print("  • Reduced false positives")
    print("  • Higher IoU for architectural elements")
    print("\n⚠️  Note: Slightly slower than Tiny model but more accurate")
    print("="*70)
