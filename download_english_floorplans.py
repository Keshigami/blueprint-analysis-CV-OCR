"""
Download Voxel51/FloorPlanCAD - English Blueprint Dataset
15,000+ real-world CAD floor plans with English annotations
"""

from datasets import load_dataset
from pathlib import Path
import requests

def download_floorplan_cad(output_dir="english_data/floorplan_cad", num_samples=200):
    """
    Download from Hugging Face: Voxel51/FloorPlanCAD
    15,000+ real-world CAD drawings in English
    """
    print(f"\n=== Downloading Voxel51/FloorPlanCAD (English) ===")
    print(f"Target: {num_samples} samples")
    print(f"Dataset: 15,000+ real-world floor plans")
    
    try:
        # Load dataset from Hugging Face
        print("Loading dataset from Hugging Face...")
        dataset = load_dataset("Voxel51/FloorPlanCAD", split="train", streaming=True)
        
        images_dir = Path(output_dir) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for i, sample in enumerate(dataset):
            if count >= num_samples:
                break
            
            try:
                # Extract image
                if 'image' in sample:
                    img = sample['image']
                    
                    # Convert to RGB if needed
                    if hasattr(img, 'convert'):
                        img = img.convert('RGB')
                    
                    # Save with high quality
                    save_path = images_dir / f"floorplan_{count:04d}.jpg"
                    img.save(save_path, quality=95)
                    
                    if count % 20 == 0:
                        print(f"✓ Downloaded {count}/{num_samples}...")
                    count += 1
                    
            except Exception as e:
                print(f"⚠ Skipping sample {i}: {e}")
                continue
        
        print(f"\n✅ Downloaded {count} English floor plans from Voxel51/FloorPlanCAD")
        print(f"📁 Saved to: {images_dir}")
        return count
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"Trying alternative: zimhe/pseudo-floor-plan-12k...")
        return download_pseudo_floorplan_alt(output_dir)


def download_pseudo_floorplan_alt(output_dir="english_data/pseudo"):
    """
    Fallback: zimhe/pseudo-floor-plan-12k
    12,000 synthetic floor plans
    """
    print(f"\n=== Fallback: zimhe/pseudo-floor-plan-12k ===")
    
    try:
        dataset = load_dataset("zimhe/pseudo-floor-plan-12k", split="train", streaming=True)
        
        images_dir = Path(output_dir) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for i, sample in enumerate(dataset):
            if count >= 200:
                break
            
            try:
                if 'image' in sample:
                    img = sample['image']
                    if hasattr(img, 'convert'):
                        img = img.convert('RGB')
                    
                    save_path = images_dir / f"pseudo_{count:04d}.jpg"
                    img.save(save_path, quality=95)
                    
                    if count % 20 == 0:
                        print(f"✓ Downloaded {count}/200...")
                    count += 1
            except Exception as e:
                continue
        
        print(f"\n✅ Downloaded {count} synthetic floor plans")
        print(f"📁 Saved to: {images_dir}")
        return count
        
    except Exception as e:
        print(f"❌ Both datasets failed: {e}")
        return 0


if __name__ == "__main__":
    print("=" * 60)
    print("English Blueprint Dataset Downloader")
    print("=" * 60)
    
    # Try primary dataset (Voxel51/FloorPlanCAD)
    count = download_floorplan_cad(num_samples=200)
    
    if count > 0:
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS: Downloaded {count} English floor plans")
        print(f"📁 Location: english_data/floorplan_cad/images/")
        print(f"🚀 Next step: python finetune_vit.py")
        print(f"{'='*60}")
    else:
        print("\n❌ No datasets available. Please download manually:")
        print("1. Visit: https://github.com/WeiliGuo/ResPlan")
        print("2. Or use synthetic data generation")
