# Installation Guide

Complete setup instructions for the Blueprint Analysis system.

## Prerequisites

### System Requirements

- **OS**: macOS, Linux, or Windows (WSL recommended)
- **Python**: 3.11 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space

### Required Software

- Python 3.11+
- Git
- pip (Python package manager)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Keshigami/blueprint-analysis-CV-OCR.git
cd blueprint_analysis
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv_py311
source venv_py311/bin/activate  # On Windows: venv_py311\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Models

#### OCR Models (Auto-download)

PaddleOCR models will download automatically on first use.

#### SAM2 Model

```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

#### Classification Model

Already included: `blueprint_classifier_english.pth`

### 5. Download Dataset (Optional)

For training or testing:

```bash
python download_english_floorplans.py
```

## Verification

### Test Installation

```bash
python -c "import torch; import paddleocr; import fastapi; print('✅ All dependencies installed!')"
```

### Run API Tests

```bash
python test_api.py
```

Expected output:

```
✅ PASS: analyze
✅ PASS: segment
✅ PASS: analyze_complete
```

### Start the Server

```bash
python app.py
```

Visit: <http://localhost:8000>

## Troubleshooting

### PaddleOCR Installation Issues

**macOS ARM (M1/M2)**:

```bash
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/mac/cpu/stable.html
pip install paddleocr
```

**Linux**:

```bash
pip install paddlepaddle
pip install paddleocr
```

### SAM2 Installation Issues

If SAM2 fails to load:

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Port 8000 Already in Use

```bash
# Kill existing process
pkill -f "python app.py"

# Or use different port
uvicorn app:app --port 8001
```

## Next Steps

- [Quick Start Guide](Quick-Start.md)
- [API Reference](API-Reference.md)
- [Training Guide](Training-Guide.md)

## Environment Variables

No environment variables required! The system works out of the box.

## Optional Enhancements

### GPU Support

For faster inference (NVIDIA GPUs only):

```bash
pip install paddlepaddle-gpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Production Deployment

For production use:

```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

**Having issues?** Check the [Troubleshooting Guide](Troubleshooting.md)
