# English Architectural Blueprint Datasets

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
