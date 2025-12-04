import torch
import numpy as np
import cv2

class SAM2Segmenter:
    """
    Production-quality segmentation using Meta's SAM2.
    """
    
    def __init__(self, checkpoint_path=None, model_cfg="sam2.1_hiera_tiny.yaml", device="cpu"):
        """
        Initialize SAM2 model.
        
        Args:
            checkpoint_path: Path to SAM2 checkpoint (.pt file)
            model_cfg: Model configuration (tiny/small/base variants)
            device: "cuda" or "cpu"
        """
        self.device = device
        
        # Download checkpoint if not provided
        if checkpoint_path is None:
            checkpoint_path = self._download_checkpoint()
        
        # Build SAM2 model
        print(f"Loading SAM2 model on {device}...")
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
       # Use SAM2ImagePredictor which handles config internally
        self.predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-tiny",
            device=device
        )
        
        print("SAM2 initialized successfully!")
    
    def _download_checkpoint(self):
        """Download SAM2 checkpoint if needed."""
        # For now, return expected path - user will need to download manually
        # SAM2 checkpoints: https://github.com/facebookresearch/segment-anything-2/tree/main/checkpoints
        checkpoint_path = "sam2.1_hiera_tiny.pt"
        print(f"Expected checkpoint at: {checkpoint_path}")
        print("Download from: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt")
        return checkpoint_path
    
    def segment(self, image_path):
        """
        Generate masks for all objects in the image.
        
        Returns:
            List of masks with bounding boxes
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image for prediction
        self.predictor.set_image(image_rgb)
        
        # For automatic segmentation, we'll sample points across the image
        h, w = image_rgb.shape[:2]
        points_per_side = 16
        step = max(h, w) // points_per_side
        
        masks_list = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                try:
                    masks, scores, _ = self.predictor.predict(
                        point_coords=np.array([[x, y]]),
                        point_labels=np.array([1]),
                        multimask_output=False
                    )
                    if len(masks) > 0 and scores[0] > 0.5:
                        # Create mask dict
                        mask = masks[0]
                        y_indices, x_indices = np.where(mask)
                        if len(y_indices) > 0:
                            bbox = [
                                int(x_indices.min()),
                                int(y_indices.min()),
                                int(x_indices.max() - x_indices.min()),
                                int(y_indices.max() - y_indices.min())
                            ]
                            masks_list.append({
                                'segmentation': mask,
                                'area': int(mask.sum()),
                                'bbox': bbox,
                                'predicted_iou': float(scores[0])
                            })
                except:
                    continue
        
        print(f"Generated {len(masks_list)} masks")
        return masks_list
    
    def visualize_masks(self, image_path, masks, output_path="segmented_output.jpg"):
        """Visualize segmentation masks on the image."""
        image = cv2.imread(image_path)
        
        # Create overlay
        overlay = image.copy()
        
        # Sort masks by area (largest first)
        masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        for i, mask_dict in enumerate(masks_sorted):
            # Random color for each mask
            color = np.random.randint(0, 255, size=3).tolist()
            
            # Get mask and ensure it's boolean
            mask = mask_dict['segmentation'].astype(bool)
            
            # Apply mask
            overlay[mask] = (overlay[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        
        # Blend original and overlay
        result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        
        # Save
        cv2.imwrite(output_path, result)
        print(f"Saved visualization to {output_path}")
        return result

if __name__ == "__main__":
    import glob
    
    # Note: This will fail until SAM2 checkpoint is downloaded
    print("\n" + "="*50)
    print("SAM2 Segmentation Test")
    print("="*50)
    print("\nIMPORTANT: Download SAM2 checkpoint first:")
    print("wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt")
    print("="*50 + "\n")
    
    try:
        # Initialize SAM2
        segmenter = SAM2Segmenter(device="cpu")
        
        # Test on a real floor plan
        test_images = glob.glob("english_data/floorplan_cad/images/*.jpg")[:1]
        
        if test_images:
            img_path = test_images[0]
            print(f"\nTesting on: {img_path}")
            
            masks = segmenter.segment(img_path)
            segmenter.visualize_masks(img_path, masks, "sam2_test_output.jpg")
            
            # Print statistics
            print(f"\nSegmentation statistics:")
            print(f"  Total masks: {len(masks)}")
            total_area = sum(m['area'] for m in masks)
            print(f"  Total covered area: {total_area} pixels")
            
    except FileNotFoundError as e:
        print(f"\n⚠️  Checkpoint not found. Please download it first:")
        print(f"wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt")
