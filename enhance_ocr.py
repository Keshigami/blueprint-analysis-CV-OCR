"""
OCR Enhancement Script
Fine-tune PaddleOCR on architectural blueprint text with custom vocabulary
"""

import os
from paddleocr import PaddleOCR
import cv2
import json
from pathlib import Path
from collections import Counter

class ArchitecturalOCREnhancer:
    """Enhanced OCR specifically for architectural blueprints"""
    
    def __init__(self):
        # Initialize with architectural optimizations
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='en',
            text_det_thresh=0.2,  # Lower threshold for faint text
            text_det_box_thresh=0.3,  # Lower box threshold
            text_recognition_batch_size=16  # Larger batch
        )
        
        # Architectural vocabulary for post-processing
        self.arch_vocab = self._load_architectural_vocabulary()
        
    def _load_architectural_vocabulary(self):
        """Load common architectural terms"""
        vocab = {
            # Room types
            'living room', 'bedroom', 'bathroom', 'kitchen', 'dining room',
            'hall', 'corridor', 'balcony', 'terrace', 'garage', 'storage',
            'closet', 'utility', 'entry', 'foyer', 'pantry', 'laundry',
            
            # Measurements
            'sqft', 'sqm', 'sq.ft', 'sq.m', 'feet', 'meters', 'inches',
            
            # Common abbreviations
            'br', 'ba', 'w/d', 'w/i', 'mstr', 'gst', 'dn', 'up',
            
            # Building elements
            'door', 'window', 'wall', 'stairs', 'elevator', 'fireplace',
            
            # Directions
            'north', 'south', 'east', 'west', 'n', 's', 'e', 'w'
        }
        return vocab
    
    def enhance_detection(self, image_path):
        """
        Enhanced OCR detection with architectural optimizations
        """
        # Load image
        img = cv2.imread(image_path)
        
        # Preprocessing for better OCR
        # 1. Increase contrast
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l,a,b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 2. Denoise
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # Run OCR on enhanced image (use predict instead of deprecated ocr)
        result = self.ocr.predict(denoised)
        
        # Post-process results
        detections = []
        if result and result[0]:
            for line in result[0]:
                box, (text, confidence) = line[0], line[1]
                
                # Validate against vocabulary
                text_lower = text.lower().strip()
                adjusted_conf = confidence
                
                # Boost confidence if matches vocabulary
                for term in self.arch_vocab:
                    if term in text_lower:
                        adjusted_conf = min(confidence + 0.15, 1.0)
                        break
                
                detections.append({
                    'text': text,
                    'confidence': adjusted_conf,
                    'original_confidence': confidence,
                    'box': box
                })
        
        return detections
    
    def batch_enhance(self, image_dir, output_file="enhanced_ocr_results.json"):
        """Process all images in directory"""
        results = {}
        images = list(Path(image_dir).glob("*.jpg"))
        
        print(f"Processing {len(images)} images with enhanced OCR...")
        for img_path in images:
            detections = self.enhance_detection(str(img_path))
            results[img_path.name] = detections
            print(f"  {img_path.name}: {len(detections)} text regions")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Saved enhanced OCR results to {output_file}")
        return results

if __name__ == "__main__":
    print("="*70)
    print("Enhanced Architectural OCR")
    print("="*70)
    
    enhancer = ArchitecturalOCREnhancer()
    
    # Test on sample images
    test_dir = "english_data/floorplan_cad/images"
    results = enhancer.batch_enhance(test_dir)
    
    # Statistics
    total_detections = sum(len(v) for v in results.values())
    avg_detections = total_detections / len(results) if results else 0
    
    print(f"\n📊 Statistics:")
    print(f"  Total images: {len(results)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average per image: {avg_detections:.1f}")
