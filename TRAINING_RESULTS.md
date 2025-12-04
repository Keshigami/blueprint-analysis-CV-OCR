# English Blueprint Model - Training Results

## Dataset

- **Source**: Voxel51/FloorPlanCAD
- **Size**: 200 real-world CAD floor plans
- **Language**: English
- **Split**: 160 train / 40 validation

## Training Configuration

- **Model**: ResNet18 (pretrained on ImageNet)
- **Epochs**: 5
- **Batch Size**: 8
- **Learning Rate**: 0.0001
- **Optimizer**: Adam

## Results

### Training Metrics

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1     | 0.1697     | 98.12%    | 0.0255   | 100.00% |
| 2     | 0.0082     | 100.00%   | 0.0034   | 100.00% |
| 3     | 0.0024     | 100.00%   | 0.0016   | 100.00% |
| 4     | 0.0017     | 100.00%   | 0.0010   | 100.00% |
| 5     | 0.0009     | 100.00%   | 0.0007   | 100.00% |

### Final Performance

- ✅ **Validation Accuracy**: 100.00%
- ✅ **Final Train Loss**: 0.0009
- ✅ **Final Val Loss**: 0.0007
- ✅ **Model**: `blueprint_classifier_english.pth`

## Comparison

| Metric | Previous (Finnish) | New (English) |
|--------|-------------------|---------------|
| Dataset | CubiCasa5k (Finnish) | Voxel51 (English) |
| Samples | 200 | 200 |
| Val Accuracy | 95.24% | **100.00%** |
| Language | Finnish text | **English text** ✓ |

## Improvements

1. **Language**: Now trained on English floor plans only
2. **Accuracy**: Improved from 95.24% to 100.00%
3. **Convergence**: Fast convergence in 5 epochs
4. **Dataset Quality**: Real-world CAD drawings vs procedural generation

## Next Steps

- ✅ Model trained and saved
- ✅ English dataset integrated
- 🔄 Pushing to GitHub...
