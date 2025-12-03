import os
import json
import glob
import cv2
import jiwer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ocr_pipeline import BlueprintOCR
from segmentation_vl import BlueprintSegmenter, VisionLanguageLabeler

def evaluate(data_dir):
    print(f"Evaluating on data in {data_dir}...")
    
    # Initialize Models
    ocr_model = BlueprintOCR(use_angle_cls=False)
    segmenter = BlueprintSegmenter()
    vl_model = VisionLanguageLabeler()
    
    # Metrics Storage
    ocr_ground_truth = []
    ocr_predictions = []
    
    cls_ground_truth = []
    cls_predictions = []
    
    files = glob.glob(os.path.join(data_dir, "*.jpg"))
    
    for file_path in files:
        json_path = file_path.replace(".jpg", ".json")
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, "r") as f:
            gt_data = json.load(f)
            
        print(f"Processing {os.path.basename(file_path)}...")
        
        # --- OCR Evaluation ---
        # Get Ground Truth Text (concatenated)
        gt_texts = [item['text'] for item in gt_data['text_regions']]
        gt_text_full = " ".join(gt_texts)
        
        # Run OCR
        ocr_results = ocr_model.process_image(file_path)
        pred_texts = [item['text'] for item in ocr_results]
        pred_text_full = " ".join(pred_texts)
        
        ocr_ground_truth.append(gt_text_full)
        ocr_predictions.append(pred_text_full)
        
        # --- Classification Evaluation ---
        # For this eval, we will cheat slightly and use the GT boxes to crop regions
        # to strictly evaluate the *Classifier* (CLIP), isolating it from Segmentation errors.
        # Alternatively, we could evaluate the full pipeline (IoU), but that's harder for this quick script.
        # Let's evaluate the Classifier on GT regions.
        
        img = cv2.imread(file_path)
        labels = ["living room", "kitchen", "bedroom", "bathroom", "door", "window"]
        
        for room in gt_data['room_labels']:
            x, y, w, h = room['bbox']
            # Ensure within bounds
            h_img, w_img = img.shape[:2]
            x = max(0, x); y = max(0, y)
            w = min(w, w_img - x); h = min(h, h_img - y)
            
            crop = img[y:y+h, x:x+w]
            if crop.size == 0: continue
            
            pred_label, score = vl_model.classify_region(crop, labels)
            
            cls_ground_truth.append(room['label'])
            cls_predictions.append(pred_label)

    # --- Calculate Metrics ---
    
    # OCR Metrics
    wer = jiwer.wer(ocr_ground_truth, ocr_predictions)
    cer = jiwer.cer(ocr_ground_truth, ocr_predictions)
    
    # Classification Metrics
    acc = accuracy_score(cls_ground_truth, cls_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(cls_ground_truth, cls_predictions, average='weighted', zero_division=0)
    
    # Report
    report = f"""
# Evaluation Report

## OCR Performance
- **WER (Word Error Rate)**: {wer:.4f}
- **CER (Character Error Rate)**: {cer:.4f}

## Classification Performance (CLIP)
- **Accuracy**: {acc:.4f}
- **Precision**: {precision:.4f}
- **Recall**: {recall:.4f}
- **F1-Score**: {f1:.4f}

## Detailed Stats
- Total Images: {len(files)}
- Total Text Regions: {len(ocr_ground_truth)} (aggregated per image)
- Total Classified Rooms: {len(cls_ground_truth)}
    """
    
    print(report)
    with open("evaluation_report.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    evaluate("eval_data")
