"""
Fine-tune PaddleOCR on English Floor Plan Dataset
Improves accuracy for architectural blueprint text recognition
"""

import os
from pathlib import Path
import cv2
import numpy as np
from paddleocr import PaddleOCR
import json

def create_ocr_training_data(image_dir="english_data/floorplan_cad/images", output_dir="ocr_training_data"):
    """
    Prepare training data for OCR fine-tuning
    Extract text regions from floor plans
    """
    print("="*70)
    print("OCR Fine-Tuning Data Preparation")
    print("="*70)
    
    # Create output directories
    Path(output_dir).mkdir(exist_ok=True)
    Path(f"{output_dir}/images").mkdir(exist_ok=True)
    Path(f"{output_dir}/labels").mkdir(exist_ok=True)
    
    # Initialize OCR for baseline extraction
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    # Process images
    images = list(Path(image_dir).glob("*.jpg"))[:50]  # Use 50 for fine-tuning
    print(f"\nProcessing {len(images)} images for OCR fine-tuning...")
    
    training_data = []
    
    for idx, img_path in enumerate(images):
        print(f"\r  Processing {idx+1}/{len(images)}...", end="")
        
        try:
            # Load and process image
            img = cv2.imread(str(img_path))
            result = ocr.ocr(str(img_path))
            
            if result and result[0]:
                # Save image
                output_img_path = f"{output_dir}/images/train_{idx:04d}.jpg"
                cv2.imwrite(output_img_path, img)
                
                # Save labels
                labels = []
                if isinstance(result[0], dict):
                    texts = result[0].get('rec_texts', [])
                    boxes = result[0].get('rec_polys', [])
                    scores = result[0].get('rec_scores', [])
                    
                    for text, box, score in zip(texts, boxes, scores):
                        if score > 0.5:  # Only high confidence
                            # Convert box to list
                            if hasattr(box, 'tolist'):
                                box = box.tolist()
                            labels.append({
                                "transcription": text,
                                "points": box,
                                "difficult": score < 0.7
                            })
                
                label_path = f"{output_dir}/labels/train_{idx:04d}.json"
                with open(label_path, 'w') as f:
                    json.dump(labels, f, indent=2)
                
                training_data.append({
                    "image": output_img_path,
                    "label": label_path,
                    "num_texts": len(labels)
                })
        
        except Exception as e:
            continue
    
    print(f"\n\n✅ Prepared {len(training_data)} training samples")
    print(f"📁 Images: {output_dir}/images/")
    print(f"📁 Labels: {output_dir}/labels/")
    
    # Save training manifest
    manifest_path = f"{output_dir}/train_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"📝 Manifest: {manifest_path}")
    
    return training_data


def fine_tune_ocr_config():
    """
    Create configuration for OCR fine-tuning
    """
    config = {
        "model": "en_PP-OCRv5_mobile_rec",
        "dataset_dir": "ocr_training_data",
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 0.0001,
        "save_dir": "finetuned_ocr_model",
        "description": "Fine-tuned on English architectural floor plans"
    }
    
    print("\n" + "="*70)
    print("OCR Fine-Tuning Configuration")
    print("="*70)
    print(json.dumps(config, indent=2))
    print("="*70)
    
    return config


def evaluate_improvements():
    """
    Evaluate OCR improvements after fine-tuning
    """
    print("\n" + "="*70)
    print("Evaluation Strategy")
    print("="*70)
    
    print("""
1. **Baseline Metrics** (Current):
   - Test on 50 English floor plans
   - Measure: Accuracy, Character Error Rate (CER)
   
2. **Post Fine-Tuning**:
   - Same 50 test images
   - Compare improvements
   
3. **Target Improvements**:
   - +5-10% accuracy on architectural terms
   - Better recognition of:
     * Room names (bedroom, kitchen, etc.)
     * Measurements (3.5m x 4.2m)
     * Annotations
""")


if __name__ == "__main__":
    print("\n🎯 OCR Fine-Tuning Pipeline")
    print("="*70)
    
    # Step 1: Prepare training data
    training_data = create_ocr_training_data()
    
    # Step 2: Configuration
    config = fine_tune_ocr_config()
    
    # Step 3: Evaluation strategy
    evaluate_improvements()
    
    print("\n" + "="*70)
    print("✅ READY FOR FINE-TUNING")
    print("="*70)
    print("""
Next Steps:
1. Review prepared training data
2. Fine-tune using PaddleOCR training API
3. Evaluate improvements
4. Deploy finetuned model

Note: Full fine-tuning requires PaddlePaddle training suite.
For now, we've prepared the data and can use the enhanced
architectural_ocr.py with domain-specific post-processing.
""")
    print("="*70)
