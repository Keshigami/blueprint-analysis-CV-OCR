# Blueprint Analysis AI

A production-ready computer vision system for architectural drawing analysis, featuring state-of-the-art segmentation, domain-specific OCR, and deep learning classification.

## 🎯 Features

### Core Capabilities

- **SAM2 Segmentation**: Meta's Segment Anything Model 2 for precise object detection
- **Architectural OCR**: Domain-optimized text recognition with multi-lingual support
- **Deep Learning Classification**: Fine-tuned ResNet18 (95.24% accuracy)
- **Vision-Language Understanding**: CLIP-based semantic labeling
- **Interactive Web Demo**: Real-time visualization at `http://localhost:8000`

### Technical Highlights

- Python 3.11 environment with PyTorch
- Trained on 100 real CubiCasa5k floor plans
- Comprehensive evaluation framework (WER/CER, Accuracy/F1)
- FastAPI production server with batch inference
- Enhanced preprocessing pipeline for architectural drawings

## 📁 Project Structure

```
blueprint_analysis/
├── sam2_segmentation.py        # SAM2 integration
├── architectural_ocr.py         # Domain-specific OCR
├── segmentation_vl.py          # CLIP classification
├── finetune_vit.py             # Model training
├── enhanced_preprocessing.py   # Image enhancement
├── app.py                      # FastAPI server
├── static/index.html           # Web demo
├── evaluate_pipeline.py        # Metrics evaluation
├── download_data.py            # Dataset downloader
├── venv_py311/                 # Python 3.11 environment
└── sam2.1_hiera_tiny.pt       # SAM2 weights (148MB)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (required for SAM2)
- Homebrew (macOS)

### Installation

1. **Clone and navigate:**

```bash
cd blueprint_analysis
```

2. **Activate Python 3.11 environment:**

```bash
source venv_py311/bin/activate
```

3. **Install dependencies** (if starting fresh):

```bash
pip install torch torchvision opencv-python pillow numpy pandas \
            paddlepaddle paddleocr transformers fastapi uvicorn \
            matplotlib scikit-learn jiwer datasets pyarrow
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

4. **Download SAM2 weights** (if not present):

```bash
curl -L -o sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

### Running the Demo

**Start the web server:**

```bash
python app.py
```

**Open browser:**
Navigate to `http://localhost:8000` and drag-drop a blueprint image to analyze.

## 🧪 Example Usage

### SAM2 Segmentation

```python
from sam2_segmentation import SAM2Segmenter

segmenter = SAM2Segmenter(device="cpu")
masks = segmenter.segment("path/to/blueprint.jpg")
segmenter.visualize_masks("path/to/blueprint.jpg", masks, "output.jpg")
```

### Architectural OCR

```python
from architectural_ocr import ArchitecturalOCR

ocr = ArchitecturalOCR()
results = ocr.process_image("path/to/blueprint.jpg")
for r in results:
    print(f"{r['text']} (confidence: {r['confidence']:.2f})")
```

### Fine-Tuned Classification

```python
import torch
from finetune_vit import finetune_model

# Train model
model = finetune_model(data_dir="real_data/images", epochs=5)
```

## 📊 Performance

| Component | Metric | Score |
|-----------|--------|-------|
| SAM2 Segmentation | Masks/Image | 185 |
| Classification | Val Accuracy | 95.24% |
| OCR (Real Data) | Confidence Range | 0.3-0.99 |
| Training Loss | Final | 0.049 |

## 🔬 Evaluation

Run comprehensive evaluation:

```bash
python evaluate_pipeline.py
```

Generates:

- OCR metrics (WER/CER)
- Classification metrics (Accuracy/F1)
- Detailed performance report

## 💾 Data

**Real-World Dataset:**

- Source: CubiCasa5k (via Hugging Face)
- Size: 100 architectural floor plans
- Download: `python download_data.py`

## 🛠️ Development

### Environment Setup

The project uses Python 3.11 for SAM2 compatibility:

```bash
# Install Python 3.11
brew install python@3.11

# Create virtual environment
/opt/homebrew/bin/python3.11 -m venv venv_py311

# Activate
source venv_py311/bin/activate
```

### Testing

```bash
# Test SAM2
python sam2_segmentation.py

# Test Architectural OCR
python architectural_ocr.py

# Test on real data
python test_real_data.py
```

## 📝 API Reference

### FastAPI Endpoints

**POST /analyze**

- Upload blueprint for analysis
- Returns: OCR results with bounding boxes
- Example:

```bash
curl -X POST -F "file=@blueprint.jpg" http://localhost:8000/analyze
```

## 🎓 Background

Developed as a portfolio project demonstrating:

- Computer Vision expertise (OCR, segmentation, classification)
- Deep Learning (PyTorch, fine-tuning, transfer learning)
- MLOps (API design, batch inference, evaluation)
- Domain adaptation (architectural notation handling)
- SOTA models (SAM2, CLIP, Vision Transformers)

## 📚 Technologies

- **Deep Learning**: PyTorch, SAM2, CLIP
- **Computer Vision**: OpenCV, PaddleOCR
- **Web Framework**: FastAPI, Uvicorn
- **Data**: Hugging Face Datasets, CubiCasa5k
- **Evaluation**: scikit-learn, jiwer

## 🤝 Contributing

This is a portfolio project. For questions or suggestions, feel free to reach out.

## 📄 License

Educational/Portfolio use.

## 🙏 Acknowledgments

- Meta AI for SAM2
- CubiCasa for the floor plan dataset
- PaddlePaddle for OCR framework
- OpenAI for CLIP model
