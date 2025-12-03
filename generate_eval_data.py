import cv2
import numpy as np
import os
import json
import random

def create_eval_sample(filename, index):
    width, height = 800, 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    ground_truth = {
        "image_file": filename,
        "text_regions": [],
        "room_labels": []
    }

    # Define some rooms and texts
    rooms = [
        {"label": "living room", "rect": (50, 50, 300, 300), "text": "LIVING ROOM", "text_pos": (100, 200)},
        {"label": "kitchen", "rect": (350, 50, 250, 200), "text": "KITCHEN", "text_pos": (400, 150)},
        {"label": "bedroom", "rect": (350, 250, 250, 250), "text": "BEDROOM", "text_pos": (400, 375)}
    ]
    
    # Randomize slightly
    for room in rooms:
        x, y, w, h = room["rect"]
        # Add noise to rect
        x += random.randint(-10, 10)
        y += random.randint(-10, 10)
        
        # Draw room
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
        
        # Draw text
        tx, ty = room["text_pos"]
        tx += random.randint(-5, 5)
        ty += random.randint(-5, 5)
        cv2.putText(img, room["text"], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add to ground truth
        ground_truth["text_regions"].append({
            "text": room["text"],
            "box": [tx, ty-20, 200, 30] # Approximate box
        })
        ground_truth["room_labels"].append({
            "label": room["label"],
            "bbox": [x, y, w, h]
        })

    # Add dimensions text (harder to map to rooms directly, but good for OCR)
    dim_text = "15' x 20'"
    cv2.putText(img, dim_text, (120, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    ground_truth["text_regions"].append({"text": dim_text, "box": [120, 215, 100, 20]})

    # Perspective transform (optional, keep simple for now to ensure OCR works well for baseline)
    # center = (width // 2, height // 2)
    # M = cv2.getRotationMatrix2D(center, random.uniform(-2, 2), 1.0)
    # img = cv2.warpAffine(img, M, (width, height), borderValue=(255, 255, 255))

    # Save Image
    cv2.imwrite(filename, img)
    
    # Save Ground Truth
    json_filename = filename.replace(".jpg", ".json")
    with open(json_filename, "w") as f:
        json.dump(ground_truth, f, indent=4)
    
    print(f"Generated {filename} and {json_filename}")

if __name__ == "__main__":
    output_dir = "eval_data"
    os.makedirs(output_dir, exist_ok=True)
    for i in range(10): # Generate 10 samples
        create_eval_sample(os.path.join(output_dir, f"eval_{i}.jpg"), i)
