import os
import cv2
from preprocessing import normalize_blueprint
from ocr_pipeline import BlueprintOCR
from generate_samples import create_synthetic_blueprint

def main():
    # 1. Generate Samples
    print("Generating samples...")
    os.makedirs("sample_blueprints", exist_ok=True)
    sample_path = "sample_blueprints/test_blueprint.jpg"
    create_synthetic_blueprint(sample_path)

    # 2. Preprocessing
    print("Running preprocessing...")
    original = cv2.imread(sample_path)
    processed = normalize_blueprint(original)
    processed_path = "sample_blueprints/test_blueprint_processed.jpg"
    cv2.imwrite(processed_path, processed)
    print(f"Saved processed image to {processed_path}")

    # 3. OCR
    print("Running OCR...")
    ocr = BlueprintOCR(use_angle_cls=False) # Disable angle cls for speed/simplicity if model not present
    results = ocr.process_image(processed_path)
    
    print("\nOCR Results:")
    print(ocr.to_json(results))

if __name__ == "__main__":
    main()
