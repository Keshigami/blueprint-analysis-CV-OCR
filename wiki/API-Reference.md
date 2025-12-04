# API Reference

Complete reference for all REST API endpoints.

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Analyze (OCR Only)

Extract text from blueprint images.

**Endpoint**: `POST /analyze`

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | Yes | Blueprint image (JPG, PNG) |
| use_architectural_ocr | Boolean | No | Use enhanced OCR (default: true) |

**Request**:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@floorplan.jpg" \
  -F "use_architectural_ocr=true"
```

**Response**:

```json
{
  "status": "success",
  "ocr_results": [
    {
      "text": "Living Room",
      "confidence": 0.95,
      "box": [[100, 50], [300, 50], [300, 80], [100, 80]]
    }
  ],
  "num_text_regions": 15,
  "ocr_type": "architectural"
}
```

### 2. Segment (SAM2 Only)

Generate segmentation masks for blueprint regions.

**Endpoint**: `POST /segment`

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | Yes | Blueprint image (JPG, PNG) |

**Request**:

```bash
curl -X POST "http://localhost:8000/segment" \
  -F "file=@floorplan.jpg"
```

**Response**:

```json
{
  "status": "success",
  "num_masks": 250,
  "masks": [
    {
      "area": 15000,
      "bbox": [50, 100, 200, 150],
      "predicted_iou": 0.85
    }
  ],
  "total_area": 450000
}
```

### 3. Analyze Complete (OCR + SAM2)

Combined analysis with both text detection and segmentation.

**Endpoint**: `POST /analyze_complete`

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | Yes | Blueprint image (JPG, PNG) |
| use_architectural_ocr | Boolean | No | Enhanced OCR (default: true) |

**Request**:

```bash
curl -X POST "http://localhost:8000/analyze_complete" \
  -F "file=@floorplan.jpg"
```

**Response**:

```json
{
  "status": "success",
  "ocr_results": [...],
  "segmentation": {
    "num_masks": 250,
    "masks": [...],
    "total_area": 450000
  }
}
```

## Response Formats

### Success Response

```json
{
  "status": "success",
  ...data fields
}
```

### Error Response

```json
{
  "status": "error",
  "message": "Error description",
  "detail": "Detailed traceback (if available)"
}
```

## HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid file) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (SAM2 not loaded) |

## Rate Limiting

No rate limiting currently implemented. For production use, consider adding rate limiting middleware.

## Code Examples

### Python

```python
import requests

# Analyze endpoint
with open('floorplan.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/analyze', files=files)
    data = response.json()
    print(f"Found {data['num_text_regions']} text regions")
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/analyze_complete', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => console.log(data));
```

### cURL

```bash
# OCR only
curl -X POST http://localhost:8000/analyze -F "file=@blueprint.jpg"

# Segmentation only  
curl -X POST http://localhost:8000/segment -F "file=@blueprint.jpg"

# Both
curl -X POST http://localhost:8000/analyze_complete -F "file=@blueprint.jpg"
```

## Best Practices

1. **Image Format**: Use JPG or PNG (JPG recommended for size)
2. **Resolution**: 1024x1024 to 2048x2048 optimal
3. **File Size**: Keep under 10MB for faster processing
4. **Timeout**: Allow 30-60 seconds for SAM2 processing

## See Also

- [API Examples](API-Examples.md) - More code samples
- [Quick Start](Quick-Start.md) - Getting started guide
- [Troubleshooting](Troubleshooting.md) - Fix common issues
