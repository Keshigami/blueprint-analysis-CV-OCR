# Dataset Summary

## Current Dataset: English Floor Plans ✅

### Source

- **Origin**: Voxel51/FloorPlanCAD
- **Type**: Real-world CAD drawings
- **Language**: English
- **Quality**: Professional architectural blueprints

### Statistics

- **Total Images**: 200
- **Format**: JPG
- **Total Size**: 28 MB
- **Location**: `english_data/floorplan_cad/images/`

### File Sizes

- **Smallest**: 24 KB (floorplan_0000.jpg)
- **Largest**: 268 KB (floorplan_0010.jpg)
- **Average**: ~143 KB per image

### Naming Convention

```
floorplan_0000.jpg
floorplan_0001.jpg
...
floorplan_0199.jpg
```

## Cleanup Summary

### Removed

- ❌ **real_data/** - 12 MB (Finnish CubiCasa5k)
- ❌ Old training data with Finnish text

### Retained

- ✅ **english_data/** - 28 MB (English Voxel51)
- ✅ 200 professional English CAD floor plans

## Usage

### View in Finder

```bash
open english_data/floorplan_cad/images/
```

### Use in Training

```python
python train_english.py  # Already trained!
```

### Test in Demo

1. Navigate to <http://localhost:8000>
2. Upload any file from `english_data/floorplan_cad/images/`
3. See OCR and SAM2 segmentation

## Sample Images

All 200 floor plans are now in:
**`/Users/keshigami/Caltech CTME/PGC AIML- ADL & Computer Vision/HeadstormAI_Job_Page/blueprint_analysis/english_data/floorplan_cad/images/`**

The Finder window has been opened showing all English floor plans! 🎉

## Dataset Credits

### FloorPlanCAD Dataset

**Attribution**:

- **Provider**: Voxel51
- **Platform**: Hugging Face Datasets
- **License**: Apache License 2.0
- **URL**: <https://huggingface.co/datasets/Voxel51/FloorPlanCAD>

**Citation**:

```bibtex
@misc{floorplancad2024,
  title={FloorPlanCAD: English Architectural Floor Plan Dataset},
  author={Voxel51},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/Voxel51/FloorPlanCAD}
}
```

**License Terms**:
This dataset is licensed under the Apache License 2.0. You may use, reproduce, and distribute this dataset in compliance with the license terms. See: <http://www.apache.org/licenses/LICENSE-2.0>

**Usage in This Project**:

- Classification model training (ResNet18, EfficientNet-B3)
- OCR text detection and recognition
- SAM2 segmentation testing
- Web demo examples

### Acknowledgment

We gratefully acknowledge Voxel51 for providing the FloorPlanCAD dataset, which has been instrumental in developing and testing this blueprint analysis system.
