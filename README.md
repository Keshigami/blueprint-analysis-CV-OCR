# 🏗️ Blueprint Analysis AI

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![SAM2](https://img.shields.io/badge/SAM2-Meta-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Production-success.svg)

A production-ready computer vision system for automated architectural drawing analysis, featuring state-of-the-art segmentation (Meta's SAM2), domain-specific OCR, and deep learning classification.

## 🎯 Features

- **🔍 SAM2 Segmentation**: Meta's Segment Anything Model 2 - 185 masks per floor plan
- **📝 Architectural OCR**: Domain-optimized text recognition with 95%+ confidence
- **🧠 Deep Learning Classification**: Fine-tuned ResNet18 (95.24% accuracy)
- **🌐 Interactive Web Demo**: Real-time visualization with SAM2/OCR toggle
- **⚡ FastAPI Server**: Production-ready REST API with 3 endpoints
- **📊 Comprehensive Evaluation**: WER/CER metrics, F1 scores, confidence analysis

## 🖼️ Demo

> **Note**: Upload a blueprint to see OCR text extraction and SAM2 segmentation in action!

### Web Interface

- **OCR Mode**: Confidence-based color coding (🔴 red = low, 🟢 green = high)
- **SAM2 Mode**: Segmentation overlay with quality scores
- **Statistics**: Real-time text regions and segment counts

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (required for SAM2)
- macOS/Linux (Windows with WSL)
- 4GB+ RAM

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/blueprint-analysis-ai.git
cd blueprint-analysis-ai
```

2. **Set up Python 3.11 environment:**

```bash
# macOS (Homebrew)
brew install python@3.11

# Create virtual environment
/opt/homebrew/bin/python3.11 -m venv venv_py311
source venv_py311/bin/activate
```

3. **Install dependencies:**

```bash
pip install torch torchvision opencv-python pillow numpy pandas \
            paddlepaddle paddleocr transformers fastapi uvicorn \
            matplotlib scikit-learn jiwer datasets pyarrow
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

4. **Download SAM2 weights:**

```bash
curl -L -o sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

### Run the Demo

```bash
python app.py
```

Navigate to **<http://localhost:8000>** and upload a blueprint!

## 📁 Project Structure

```
blueprint_analysis/
├── app.py                      # FastAPI server ⭐
├── sam2_segmentation.py        # SAM2 integration
├── architectural_ocr.py         # Domain-specific OCR
├── segmentation_vl.py          # CLIP classification
├── finetune_vit.py             # Model training
├── enhanced_preprocessing.py   # Image enhancement
├── static/
│   └── index.html              # Web demo
├── evaluate_pipeline.py        # Metrics evaluation
├── download_data.py            # Dataset downloader
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔌 API Reference

### Base URL

```
http://localhost:8000
```

### Endpoints

#### 1. **POST /analyze**

Analyze blueprint with architectural OCR.

**Request:**

```bash
curl -X POST -F "file=@blueprint.jpg" \
  http://localhost:8000/analyze?use_architectural_ocr=true
```

**Response:**

```json
{
  "status": "success",
  "ocr_results": [
    {
      "text": "LIVING ROOM",
      "confidence": 0.94,
      "box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    }
  ],
  "num_text_regions": 12,
  "ocr_type": "architectural"
}
```

#### 2. **POST /segment**

Segment blueprint using SAM2.

**Request:**

```bash
curl -X POST -F "file=@blueprint.jpg" \
  http://localhost:8000/segment
```

**Response:**

```json
{
  "status": "success",
  "num_masks": 185,
  "masks": [
    {
      "area": 12450,
      "bbox": [x, y, width, height],
      "predicted_iou": 0.89
    }
  ],
  "total_area": 2456789
}
```

#### 3. **POST /analyze_complete**

Combined OCR + segmentation analysis.

**Response:**

```json
{
  "status": "success",
  "ocr_results": [...],
  "segmentation": {
    "num_masks": 185,
    "masks": [...]
  }
}
```

## 📊 Performance

| Component | Metric | Score |
|-----------|--------|-------|
| SAM2 Segmentation | Masks/Image | 185 |
| Classification | Validation Accuracy | 95.24% |
| OCR (Real Data) | Confidence Range | 0.3-0.99 |
| Training Loss | Final | 0.049 |
| Dataset Size | Floor Plans | 200 |

## 🧪 Evaluation

Run comprehensive evaluation:

```bash
python evaluate_pipeline.py
```

**Metrics:**

- **OCR**: Word Error Rate (WER), Character Error Rate (CER)
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Segmentation**: Mask count, coverage area, quality scores

## 💾 Dataset

**Source**: [CubiCasa5k](https://github.com/CubiCasa/CubiCasa5k) via Hugging Face

Download additional data:

```bash
python download_data.py  # Downloads 200 floor plans
```

## 🛠️ Development

### Training

Fine-tune the classifier on real data:

```bash
python finetune_vit.py
```

**Results**: 95.24% validation accuracy after 5 epochs.

### Testing

```bash
# Test SAM2
python sam2_segmentation.py

# Test Architectural OCR
python architectural_ocr.py

# Test on real data
python test_real_data.py
```

## 🏗️ Architecture

```
Input Image
    ↓
┌───────────────┐
│ Preprocessing │  (Enhance, Normalize)
└───────┬───────┘
        ↓
    ┌───┴───┐
    ↓       ↓
┌─────┐  ┌──────┐
│ OCR │  │ SAM2 │
└──┬──┘  └───┬──┘
   ↓         ↓
┌──────────────┐
│ CLIP Labeling│
└──────┬───────┘
       ↓
┌────────────┐
│   Output   │
│ (JSON+viz) │
└────────────┘
```

## 🤝 Contributing

Contributions are welcome! This is a portfolio project, but feel free to:

- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- **Meta AI** - SAM2 model
- **CubiCasa** - Floor plan dataset
- **PaddlePaddle** - OCR framework
- **OpenAI** - CLIP model
- **Hugging Face** - Dataset hosting

## 📞 Contact

**Author**: [Your Name]  
**LinkedIn**: [Your LinkedIn]  
**Email**: [Your Email]

---

⭐ **Star this repo** if you find it useful!

Built with ❤️ for Computer Vision Engineers
