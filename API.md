# API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

No authentication required for local deployment.

## Endpoints

### 1. GET /

**Description**: Serves the interactive web demo.

**Response**: HTML page

---

### 2. POST /analyze

**Description**: Analyze blueprint with OCR (standard or architectural).

**Parameters:**

- `file` (required): Blueprint image file (JPG, PNG)
- `use_architectural_ocr` (optional, default=true): Use domain-specific OCR

**Request Example:**

```bash
curl -X POST \
  -F "file=@/path/to/blueprint.jpg" \
  -F "use_architectural_ocr=true" \
  http://localhost:8000/analyze
```

**Response:**

```json
{
  "status": "success",
  "ocr_results": [
    {
      "text": "KITCHEN",
      "confidence": 0.94,
      "box": [[100, 200], [300, 200], [300, 250], [100, 250]]
    }
  ],
  "num_text_regions": 8,
  "ocr_type": "architectural"
}
```

**Response Fields:**

- `status`: "success" or "error"
- `ocr_results`: Array of detected text regions
  - `text`: Detected text string
  - `confidence`: OCR confidence score (0-1)
  - `box`: Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
- `num_text_regions`: Total number of detected text regions
- `ocr_type`: "architectural" or "standard"

---

### 3. POST /segment

**Description**: Segment blueprint using SAM2.

**Parameters:**

- `file` (required): Blueprint image file

**Request Example:**

```bash
curl -X POST \
  -F "file=@/path/to/blueprint.jpg" \
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
      "bbox": [150, 200, 80, 60],
      "predicted_iou": 0.89
    }
  ],
  "total_area": 2456789
}
```

**Response Fields:**

- `status`: "success" or "error"
- `num_masks`: Total number of segmented regions
- `masks`: Array of segmentation masks (limited to 50)
  - `area`: Mask area in pixels
  - `bbox`: Bounding box [x, y, width, height]
  - `predicted_iou`: Quality score (0-1)
- `total_area`: Total segmented area in pixels

---

### 4. POST /analyze_complete

**Description**: Complete analysis with both OCR and segmentation.

**Parameters:**

- `file` (required): Blueprint image file

**Request Example:**

```bash
curl -X POST \
  -F "file=@/path/to/blueprint.jpg" \
  http://localhost:8000/analyze_complete
```

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

## Error Handling

All endpoints return errors in this format:

```json
{
  "status": "error",
  "message": "Error description"
}
```

**Common HTTP Status Codes:**

- `200`: Success
- `500`: Internal server error
- `503`: Service unavailable (e.g., SAM2 not loaded)

## Rate Limiting

No rate limiting for local deployment.

## Examples

### Python

```python
import requests

url = "http://localhost:8000/analyze_complete"
files = {"file": open("blueprint.jpg", "rb")}

response = requests.post(url, files=files)
data = response.json()

print(f"OCR results: {len(data['ocr_results'])} text regions")
print(f"Segmentation: {data['segmentation']['num_masks']} masks")
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('/analyze_complete', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => {
  console.log(`Found ${data.ocr_results.length} text regions`);
  console.log(`Found ${data.segmentation.num_masks} segments`);
});
```

## Performance

**Typical Response Times:**

- `/analyze`: 2-4 seconds
- `/segment`: 30-60 seconds (CPU), 5-10 seconds (GPU)
- `/analyze_complete`: 35-65 seconds (CPU)

**Recommendations:**

- Use GPU for production deployments
- Consider caching for repeated analyses
- Implement async processing for large batches
