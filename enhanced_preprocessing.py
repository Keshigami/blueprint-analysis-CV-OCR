import cv2
import numpy as np

def enhance_for_ocr(image):
    """
    Enhanced preprocessing specifically for OCR on architectural drawings.
    Returns processed image with improved text clarity.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # 2. Adaptive thresholding for varied lighting
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 3. Morphological operations to enhance text
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 4. Invert if background is dark
    if np.mean(morph) < 127:
        morph = cv2.bitwise_not(morph)
    
    # 5. Sharpen text
    kernel_sharp = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
    sharpened = cv2.filter2D(morph, -1, kernel_sharp)
    
    return sharpened

def auto_rotate_text(image):
    """
    Detect and correct text rotation for better OCR.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)
        
        # Get median angle
        median_angle = np.median(angles)
        
        # Rotate if needed (only small corrections)
        if abs(median_angle) > 1 and abs(median_angle) < 45:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), 
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
            return rotated
    
    return image

if __name__ == "__main__":
    # Test the enhancement
    import glob
    from ocr_pipeline import BlueprintOCR
    
    ocr = BlueprintOCR(use_angle_cls=False)
    test_images = glob.glob("real_data/images/*.jpg")[:3]
    
    print("Testing enhanced preprocessing...")
    for img_path in test_images:
        img = cv2.imread(img_path)
        
        # Original OCR
        results_original = ocr.process_image(img_path)
        avg_conf_original = np.mean([r['confidence'] for r in results_original]) if results_original else 0
        
        # Enhanced OCR
        enhanced = enhance_for_ocr(img)
        enhanced_path = img_path.replace(".jpg", "_enhanced.jpg")
        cv2.imwrite(enhanced_path, enhanced)
        results_enhanced = ocr.process_image(enhanced_path)
        avg_conf_enhanced = np.mean([r['confidence'] for r in results_enhanced]) if results_enhanced else 0
        
        print(f"\n{img_path}:")
        print(f"  Original avg confidence: {avg_conf_original:.3f}")
        print(f"  Enhanced avg confidence: {avg_conf_enhanced:.3f}")
        print(f"  Improvement: {(avg_conf_enhanced - avg_conf_original):.3f}")
