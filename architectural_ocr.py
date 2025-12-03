from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
from enhanced_preprocessing import enhance_for_ocr

class ArchitecturalOCR:
    """
    Domain-specific OCR optimized for architectural drawings.
    Includes preprocessing, post-processing, and architectural vocabulary.
    """
    
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        # Common architectural terms for validation
        self.architectural_vocab = {
            'living', 'kitchen', 'bedroom', 'bathroom', 'wc',
            'hall', 'corridor', 'balcony', 'terrace', 'garage',
            'storage', 'closet', 'utility', 'dining', 'entry',
            # Measurement terms
            'm', 'cm', 'mm', 'm2', 'sq', 'ft',
            # Finnish common terms (from CubiCasa)
            'keittion', 'makuuhuone', 'olohuone', 'kylpyhuone'
        }
    
    def preprocess_for_architecture(self, image):
        """Enhanced preprocessing specific to architectural drawings."""
        # Use our enhanced preprocessing
        enhanced = enhance_for_ocr(image)
        # Convert back to BGR if needed (PaddleOCR expects BGR)
        if len(enhanced.shape) == 2:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced
    
    def postprocess_text(self, text, confidence):
        """
        Post-process OCR text with architectural domain knowledge.
        """
        # 1. Clean up common OCR mistakes
        text = text.strip()
        
        # 2. Handle measurement notation
        # Fix common confusions: O -> 0, l -> 1, etc.
        if re.search(r'\d', text):  # If contains numbers
            text = text.replace('O', '0').replace('o', '0')
            text = text.replace('l', '1').replace('I', '1')
        
        # 3. Validate against architectural vocabulary
        text_lower = text.lower()
        for term in self.architectural_vocab:
            if term in text_lower and confidence < 0.7:
                # Boost confidence if matches known term
                confidence = min(confidence + 0.2, 1.0)
        
        # 4. Handle dimension formats (e.g., "3.5m x 4.2m")
        dimension_pattern = r'[\d.]+\s*[xX×]\s*[\d.]+'
        if re.search(dimension_pattern, text):
            confidence = min(confidence + 0.15, 1.0)
        
        return text, confidence
    
    def process_image(self, image_path_or_array):
        """
        Process architectural drawing with domain-specific optimizations.
        """
        # Step 1: Load image
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
        else:
            img = image_path_or_array
        
        # Step 2: Enhanced preprocessing
        processed = self.preprocess_for_architecture(img)
        
        # Step 3: Run OCR
        result = self.ocr.ocr(processed)
        
        # Step 4: Parse and post-process
        output_data = []
        if result and len(result) > 0:
            if isinstance(result[0], dict):
                # Dictionary format
                texts = result[0].get('rec_texts', [])
                scores = result[0].get('rec_scores', [])
                boxes = result[0].get('rec_polys', [])
                
                for text, score, box in zip(texts, scores, boxes):
                    # Post-process
                    cleaned_text, adjusted_conf = self.postprocess_text(text, score)
                    
                    if hasattr(box, 'tolist'):
                        box = box.tolist()
                    
                    output_data.append({
                        "text": cleaned_text,
                        "confidence": adjusted_conf,
                        "box": box,
                        "original_text": text,
                        "original_confidence": score
                    })
        
        return output_data

if __name__ == "__main__":
    import glob
    
    # Test on real data
    ocr = ArchitecturalOCR()
    test_images = glob.glob("real_data/images/*.jpg")[:3]
    
    print("Testing Architectural OCR...")
    for img_path in test_images:
        print(f"\n{img_path}:")
        results = ocr.process_image(img_path)
        
        for r in results[:5]:  # Show first 5
            improvement = r['confidence'] - r['original_confidence']
            print(f"  '{r['text']}' (conf: {r['confidence']:.3f}, +{improvement:.3f})")
