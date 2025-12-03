from datasets import load_dataset
import os
import cv2
import numpy as np

def download_real_data(output_dir="real_data", num_samples=200):
    print(f"Downloading {num_samples} samples from jprve/FloorPlansV2...")
    
    # Load dataset (streaming mode to avoid downloading everything if we only want a subset)
    # Note: 'jprve/FloorPlansV2' might be large, so streaming is safer for a demo.
    try:
        dataset = load_dataset("jprve/FloorPlansV2", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    count = 0
    for i, sample in enumerate(dataset):
        if count >= num_samples:
            break
            
        try:
            # The dataset structure depends on the specific HF repo.
            # Usually 'image' key contains a PIL image.
            if 'image' in sample:
                img = sample['image']
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save
                save_path = os.path.join(images_dir, f"real_{count}.jpg")
                img.save(save_path)
                if count % 20 == 0:
                    print(f"Downloaded {count}/{num_samples} images...")
                count += 1
        except Exception as e:
            print(f"Skipping sample {i} due to error: {e}")

    print(f"Download complete. Saved {count} images.")

if __name__ == "__main__":
    download_real_data()
