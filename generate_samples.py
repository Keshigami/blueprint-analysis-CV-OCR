import cv2
import numpy as np
import os
import random

def create_synthetic_blueprint(filename, width=800, height=600):
    # White background
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw some "rooms" (rectangles)
    # Room 1
    cv2.rectangle(img, (50, 50), (350, 350), (0, 0, 0), 2)
    cv2.putText(img, "LIVING ROOM", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "15' x 20'", (120, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Room 2
    cv2.rectangle(img, (350, 50), (600, 250), (0, 0, 0), 2)
    cv2.putText(img, "KITCHEN", (400, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Room 3
    cv2.rectangle(img, (350, 250), (600, 500), (0, 0, 0), 2)
    cv2.putText(img, "BEDROOM", (400, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Add some "noise" or perspective to make it interesting for preprocessing
    # Let's rotate it slightly
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, 2, 1.0) # 2 degrees rotation
    img = cv2.warpAffine(img, M, (width, height), borderValue=(255, 255, 255))

    cv2.imwrite(filename, img)
    print(f"Generated {filename}")

if __name__ == "__main__":
    os.makedirs("sample_blueprints", exist_ok=True)
    for i in range(3):
        create_synthetic_blueprint(f"sample_blueprints/blueprint_{i}.jpg")
