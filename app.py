from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import shutil
import os
import sys
import cv2
import numpy as np
from preprocessing import normalize_blueprint
from ocr_pipeline import BlueprintOCR
from architectural_ocr import ArchitecturalOCR

# Add venv to path for SAM2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'venv_py311/lib/python3.11/site-packages'))

app = FastAPI(title="Blueprint Analysis API")

# Mount static files
app.mount("/static", StaticFiles(directory="blueprint_analysis/static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('blueprint_analysis/static/index.html')

# Initialize models
print("Initializing OCR models...")
ocr_standard = BlueprintOCR(use_angle_cls=False)
ocr_architectural = ArchitecturalOCR()

# Initialize SAM2 (lazy loading for better startup time)
sam2_model = None

def get_sam2():
    global sam2_model
    if sam2_model is None:
        try:
            from sam2_segmentation import SAM2Segmenter
            print("Loading SAM2 model...")
            sam2_model = SAM2Segmenter(device="cpu")
            print("SAM2 loaded successfully!")
        except Exception as e:
            print(f"SAM2 initialization failed: {e}")
            sam2_model = False  # Mark as failed
    return sam2_model if sam2_model is not False else None

@app.post("/analyze")
async def analyze_blueprint(file: UploadFile = File(...), use_architectural_ocr: bool = True):
    """
    Analyze blueprint with OCR.
    
    Args:
        file: Uploaded blueprint image
        use_architectural_ocr: Use domain-specific OCR (default: True)
    
    Returns:
        JSON with OCR results and bounding boxes
    """
    # Save uploaded file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Preprocess
        img = cv2.imread(temp_path)
        processed = normalize_blueprint(img)
        processed_path = f"processed_{file.filename}"
        cv2.imwrite(processed_path, processed)
        
        # Run OCR
        ocr_engine = ocr_architectural if use_architectural_ocr else ocr_standard
        results = ocr_engine.process_image(processed_path)
        
        # Cleanup
        os.remove(temp_path)
        os.remove(processed_path)
        
        return {
            "status": "success",
            "ocr_results": results,
            "num_text_regions": len(results),
            "ocr_type": "architectural" if use_architectural_ocr else "standard"
        }
    
    except Exception as e:
        # Cleanup on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(processed_path):
            os.remove(processed_path)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/segment")
async def segment_blueprint(file: UploadFile = File(...)):
    """
    Segment blueprint using SAM2.
    
    Returns:
        JSON with segmentation masks and statistics
    """
    sam2 = get_sam2()
    if sam2 is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "SAM2 not available"}
        )
    
    # Save uploaded file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Run segmentation
        masks = sam2.segment(temp_path)
        
        # Convert masks to serializable format
        masks_serializable = []
        for mask in masks:
            masks_serializable.append({
                "area": int(mask["area"]),
                "bbox": [int(x) for x in mask["bbox"]],
                "predicted_iou": float(mask["predicted_iou"])
            })
        
        # Cleanup
        os.remove(temp_path)
        
        return {
            "status": "success",
            "num_masks": len(masks_serializable),
            "masks": masks_serializable[:50],  # Limit to 50 for response size
            "total_area": sum(m["area"] for m in masks_serializable)
        }
    
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/analyze_complete")
async def analyze_complete(file: UploadFile = File(...)):
    """
    Complete analysis: OCR + Segmentation.
    
    Returns:
        JSON with both OCR and segmentation results
    """
    # Save uploaded file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Run OCR
        img = cv2.imread(temp_path)
        processed = normalize_blueprint(img)
        processed_path = f"processed_{file.filename}"
        cv2.imwrite(processed_path, processed)
        
        ocr_results = ocr_architectural.process_image(processed_path)
        os.remove(processed_path)
        
        # Run Segmentation
        sam2 = get_sam2()
        masks = []
        if sam2:
            try:
                seg_masks = sam2.segment(temp_path)
                masks = [{
                    "area": int(m["area"]),
                    "bbox": [int(x) for x in m["bbox"]],
                    "predicted_iou": float(m["predicted_iou"])
                } for m in seg_masks[:50]]
            except:
                pass
        
        # Cleanup
        os.remove(temp_path)
        
        return {
            "status": "success",
            "ocr_results": ocr_results,
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        # Keep processed file for inspection or remove it

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
