# Frequently Asked Questions (FAQ)

## General Questions

### What is Blueprint Analysis?

A comprehensive AI system for analyzing architectural floor plans using OCR (text detection), classification, and segmentation (SAM2).

### What file formats are supported?

- **Input**: JPG, PNG
- **Recommended**: JPG for better compression

### Is this free to use?

Yes! Licensed under MIT. See [LICENSE](../LICENSE) for details.

## Installation & Setup

### Do I need a GPU?

No. The system works on CPU. GPU is optional for faster inference.

### Why is SAM2 slow on my machine?

SAM2 is computationally intensive. Expected times:

- **CPU**: 15-30s per image
- **GPU**: 3-5s per image

Consider using the Tiny model (default) instead of Base for faster results.

### Can I run this on Windows?

Yes, but WSL (Windows Subsystem for Linux) is recommended for better compatibility.

## Usage Questions

### Why is OCR not detecting all text?

Common reasons:

1. **Low resolution**: Use images >1024px
2. **Faint text**: Try preprocessing in `enhance_ocr.py`
3. **Non-English text**: System is optimized for English

### How do I improve OCR accuracy?

1. Use higher resolution images
2. Enable `use_architectural_ocr=true`
3. Preprocess with CLAHE enhancement
4. Fine-tune on your specific blueprint style

### Why are bounding boxes misaligned?

This was a known issue (now fixed). If you still see it:

1. Hard refresh: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+F5` (Windows)
2. Clear browser cache
3. Restart the server

### Can I process multiple images at once?

The API processes one image per request. For batch processing, use the API in a loop:

```python
for image in images:
    response = requests.post(url, files={'file': open(image, 'rb')})
```

## Model Questions

### What models are used?

1. **Classification**: ResNet18 (baseline) or EfficientNet-B3 (enhanced)
2. **OCR**: PaddleOCR (en_PP-OCRv5)
3. **Segmentation**: SAM2 Hiera Tiny

### Can I use my own models?

Yes! Replace the checkpoint files:

- Classification: `blueprint_classifier_english.pth`
- SAM2: `sam2.1_hiera_tiny.pt`

### How do I retrain the models?

See [Training Guide](Training-Guide.md) for detailed instructions.

```bash
python train_enhanced.py  # Classification
python finetune_ocr.py    # OCR data prep
```

## Dataset Questions

### What dataset is included?

200 English CAD floor plans from Voxel51/FloorPlanCAD.

### Can I add more data?

Yes! Options:

1. Download WAFFLE (20K images): `python download_waffle.py`
2. Use your own: Place in `custom_data/` and modify training scripts

### Is the data English-only?

Yes. Finnish data (CubiCasa5k) was removed to focus on English.

## Performance Questions

### Why is the first request slow?

Models load on first use (lazy loading):

- **PaddleOCR**: ~5s to load
- **SAM2**: ~10s to load

Subsequent requests are faster.

### How can I speed up processing?

1. **Use GPU**: Install `paddlepaddle-gpu` and CUDA
2. **Reduce image size**: Resize to 1024x1024
3. **Use baseline model**: ResNet18 is 95x faster than EfficientNet-B3
4. **Disable SAM2**: Use `/analyze` instead of `/analyze_complete`

### Can I deploy this in production?

Yes. Recommendations:

- Use gunicorn: `gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker`
- Add nginx reverse proxy
- Implement rate limiting
- Use Redis for caching

## Troubleshooting

### Port 8000 is already in use

```bash
pkill -f "python app.py"
# Or use different port
python app.py --port 8001
```

### PaddleOCR fails to import

```bash
pip uninstall paddleocr paddlepaddle
pip install paddlepaddle paddleocr
```

### SAM2 out of memory

1. Use Tiny model (default)
2. Reduce image resolution
3. Close other applications
4. Increase system swap/virtual memory

### "Image not loaded for scaling" error

This is fixed in the latest version. Update:

```bash
git pull origin main
```

Hard refresh the browser: `Cmd+Shift+R`

## Contributing

### How can I contribute?

See [Contributing Guide](Contributing.md) for:

- Bug reports
- Feature requests
- Pull request guidelines

### I found a bug, what do I do?

1. Check [existing issues](https://github.com/Keshigami/blueprint-analysis-CV-OCR/issues)
2. Create new issue with:
   - Steps to reproduce
   - Expected vs actual behavior
   - System info (OS, Python version)

## Contact

### Where can I get help?

1. Search this wiki
2. Check [Troubleshooting Guide](Troubleshooting.md)
3. [Open an issue on GitHub](https://github.com/Keshigami/blueprint-analysis-CV-OCR/issues)

---

**Still have questions?** Open an issue: <https://github.com/Keshigami/blueprint-analysis-CV-OCR/issues>
