import os
import glob
from ocr_pipeline import BlueprintOCR

def test_real_data():
    ocr = BlueprintOCR(use_angle_cls=False)
    images = glob.glob("real_data/images/*.jpg")[:5]  # Test on first 5
    
    print(f"Testing OCR on {len(images)} real floor plans...")
    for img_path in images:
        print(f"\nProcessing {os.path.basename(img_path)}...")
        results = ocr.process_image(img_path)
        print(f"Found {len(results)} text regions:")
        for r in results[:3]:  # Show first 3
            print(f"  - {r['text']} (conf: {r['confidence']:.2f})")

if __name__ == "__main__":
    test_real_data()
