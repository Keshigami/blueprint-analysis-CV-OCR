"""
Download English Architectural Blueprint Datasets
Replaces Finnish CubiCasa5k with English-language floor plans
"""

from datasets import load_dataset
import os
import PIL.Image
import requests
from pathlib import Path

def download_pseudo_floorplan_12k(output_dir="english_data/pseudo_12k", num_samples=200):
    """
    Download from Hugging Face: pseudo-floor-plan-12k dataset
    12,000 procedurally generated English floor plans
    """
    print(f"\n=== Downloading Pseudo Floor Plan 12k (English) ===")
    print(f"Target: {num_samples} samples")
    
    try:
        # Load from Hugging Face
        dataset = load_dataset("TrainingDataPro/pseudo-floor-plan-12k", 
                              split="train", 
                              streaming=True)
        
        images_dir = Path(output_dir) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for i, sample in enumerate(dataset):
            if count >= num_samples:
                break
            
            try:
                if 'image' in sample:
                    img = sample['image']
                    if hasattr(img, 'convert'):
                        img = img.convert('RGB')
                    
                    save_path = images_dir / f"english_{count}.jpg"
                    img.save(save_path, quality=95)
                    
                    if count % 20 == 0:
                        print(f"Downloaded {count}/{num_samples}...")
                    count += 1
            except Exception as e:
                print(f"Skipping sample {i}: {e}")
        
        print(f"✅ Downloaded {count} English floor plans")
        return count
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0


def download_resplan_subset(output_dir="english_data/resplan", num_samples=100):
    """
    ResPlan: 17,000 residential floor plans with English annotations
    Note: May require special access/download
    """
    print(f"\n=== Attempting ResPlan Download ===")
    print("Note: This dataset may require manual download")
    print("Visit: https://github.com/WeiliGuo/ResPlan")
    
    # Create placeholder directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {output_dir}")
    print("Please download manually from GitHub and place in this directory")
    return 0


def download_robin_dataset(output_dir="english_data/robin"):
    """
    ROBIN: 510 black & white floor plans (English)
    Small dataset but good quality
    """
    print(f"\n=== ROBIN Dataset ===")
    print("GitHub: https://github.com/gesstalt/ROBIN")
    
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory: {output_dir}")
    print("Please clone: git clone https://github.com/gesstalt/ROBIN.git")
    return 0


def create_english_dataset_info(output_dir="english_data"):
    """Create README with dataset information"""
    readme = """# English Architectural Blueprint Datasets

## Sources

### 1. Pseudo Floor Plan 12k (Primary)
- **Size**: 12,000 procedurally generated images
- **Language**: English
- **Source**: Hugging Face - TrainingDataPro/pseudo-floor-plan-12k
- **Format**: Synthetic, clean floor plans
- **Status**: ✅ Downloaded automatically

### 2. ResPlan
- **Size**: 17,000 residential floor plans
- **Language**: English  
- **Source**: https://github.com/WeiliGuo/ResPlan
- **Format**: Detailed with precise annotations
- **Status**: ⚠️ Requires manual download

### 3. ROBIN
- **Size**: 510 images
- **Language**: English
- **Source**: https://github.com/gesstalt/ROBIN
- **Format**: Black & white floor plans
- **Status**: ⚠️ Requires manual download

## Usage

1. Run this script to download Pseudo Floor Plan 12k
2. Manually download ResPlan and ROBIN if needed
3. Use for training English-specific OCR model

## Training

```bash
python finetune_vit.py --data_dir english_data/pseudo_12k/images
```
"""
    
    readme_path = Path(output_dir) / "README.md"
    readme_path.write_text(readme)
    print(f"\n✅ Created {readme_path}")


if __name__ == "__main__":
    print("English Blueprint Dataset Downloader")
    print("=====================================\n")
    
    # Create base directory
    Path("english_data").mkdir(exist_ok=True)
    
    # 1. Download primary dataset (Pseudo Floor Plan 12k)
    count = download_pseudo_floorplan_12k(num_samples=200)
    
    # 2. Info about additional datasets
    download_resplan_subset()
    download_robin_dataset()
    
    # 3. Create documentation
    create_english_dataset_info()
    
    print(f"\n{'='*50}")
    print(f"✅ Downloaded {count} English floor plans")
    print(f"📁 Location: english_data/pseudo_12k/images/")
    print(f"📝 Next: python finetune_vit.py --data english_data/pseudo_12k/images")
    print(f"{'='*50}")
