import cv2
import numpy as np

def load_image(image_path):
    """Loads an image from a file path."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return img

def to_grayscale(image):
    """Converts an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def detect_edges(image, low_threshold=50, high_threshold=150):
    """Detects edges using Canny edge detection."""
    if len(image.shape) == 3:
        image = to_grayscale(image)
    return cv2.Canny(image, low_threshold, high_threshold)

def find_contours(edge_image):
    """Finds contours in an edge-detected image."""
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_largest_contour(contours):
    """Returns the largest contour by area."""
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def order_points(pts):
    """Orders coordinates: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Applies a perspective transform to obtain a top-down view."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def normalize_blueprint(image):
    """
    Full pipeline: Grayscale -> Edge -> Contour -> Perspective Correct (if applicable).
    If no clear document contour is found, returns the original (or grayscaled) image.
    """
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = detect_edges(blurred)
    contours = find_contours(edges)
    largest = get_largest_contour(contours)

    if largest is not None:
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) == 4:
            warped = four_point_transform(image, approx.reshape(4, 2))
            return warped
    
    return image # Fallback
