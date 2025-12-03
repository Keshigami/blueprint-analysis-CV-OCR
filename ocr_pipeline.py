from paddleocr import PaddleOCR
import json

class BlueprintOCR:
    def __init__(self, lang='en', use_angle_cls=True):
        # Initialize PaddleOCR
        # use_angle_cls=True enables orientation classification
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    def process_image(self, image_path_or_array):
        """
        Runs OCR on an image (path or numpy array).
        Returns a list of dictionaries with 'text', 'confidence', and 'box'.
        """
        result = self.ocr.ocr(image_path_or_array)
        output_data = []
        if result and len(result) > 0 and isinstance(result[0], dict):
            # Handle dictionary output format (newer PaddleOCR/PaddleX)
            data = result[0]
            texts = data.get('rec_texts', [])
            scores = data.get('rec_scores', [])
            boxes = data.get('rec_polys', [])
            
            for text, score, box in zip(texts, scores, boxes):
                # Convert numpy array box to list for JSON serialization
                if hasattr(box, 'tolist'):
                    box = box.tolist()
                output_data.append({
                    "text": text,
                    "confidence": score,
                    "box": box
                })
        elif result and result[0]:
            # Handle legacy list of lists format
            for line in result[0]:
                box = line[0]
                text, confidence = line[1]
                output_data.append({
                    "text": text,
                    "confidence": confidence,
                    "box": box
                })
        return output_data

    def to_json(self, data):
        """Converts the structured data to JSON string."""
        return json.dumps(data, indent=4)

if __name__ == "__main__":
    # Test stub
    pass
