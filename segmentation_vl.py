import cv2
import torch
import numpy as np
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Placeholder for SAM2 or similar segmentation model
# In a real scenario, you would import: from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
class BlueprintSegmenter:
    def __init__(self):
        # Initialize SAM2 or fallback to contour-based segmentation for this demo
        print("Initializing Segmentation Model (Simulating SAM2)...")
    
    def segment(self, image_path):
        """
        Returns a list of masks/regions.
        For this demo, we use the contour detection from Hour 1 as a lightweight 'segmentation'.
        """
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > 500: # Filter small noise
                x, y, w, h = cv2.boundingRect(cnt)
                roi = img[y:y+h, x:x+w]
                regions.append({
                    "id": i,
                    "bbox": [x, y, w, h],
                    "crop": roi
                })
        return regions

class VisionLanguageLabeler:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        print(f"Loading Vision-Language Model ({model_name})...")
        try:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        except Exception as e:
            print(f"Warning: Could not load CLIP model ({e}). Using dummy labeler.")
            self.model = None

    def classify_region(self, region_crop, candidate_labels):
        """
        Uses CLIP to classify the region crop into one of the candidate labels.
        """
        if self.model is None:
            # Dummy fallback
            return candidate_labels[0], 0.99

        image = Image.fromarray(cv2.cvtColor(region_crop, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        best_idx = probs.argmax().item()
        return candidate_labels[best_idx], probs[0][best_idx].item()

def run_pipeline(image_path):
    # 1. Segmentation
    segmenter = BlueprintSegmenter()
    regions = segmenter.segment(image_path)
    print(f"Found {len(regions)} regions.")

    # 2. Vision-Language Classification
    vl_model = VisionLanguageLabeler()
    labels = ["living room", "kitchen", "bedroom", "bathroom", "door", "window", "text block", "floor plan"]

    results = []
    for region in regions:
        label, score = vl_model.classify_region(region['crop'], labels)
        results.append({
            "region_id": region['id'],
            "bbox": region['bbox'],
            "label": label,
            "score": score
        })
        print(f"Region {region['id']}: {label} ({score:.2f})")
    
    return results

if __name__ == "__main__":
    # Use the processed image from Hour 1
    image_path = "sample_blueprints/test_blueprint_processed.jpg"
    if not os.path.exists(image_path):
        # Fallback to generating one if not exists
        from generate_samples import create_synthetic_blueprint
        create_synthetic_blueprint(image_path)
        
    run_pipeline(image_path)
