import cv2
import numpy as np
from PIL import Image as PILImage

def enhance_historical_text_image(pil_image):
    """
    Advanced preprocessing sequence designed to eliminate bleed-through text, 
    background noise granules, and artificially boost text contrast.
    """
    # 1. Convert PIL Image to OpenCV grayscale format
    open_cv_image = np.array(pil_image)
    if len(open_cv_image.shape) == 3:
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = open_cv_image

    # 2. Apply Bilateral Filter to smooth out paper grain noise while preserving sharp text edges
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. Apply Adaptive Thresholding (Gaussian) to clean complex illumination and dark shadows
    # Block size 21 or 25 is suitable for fine tools dictionary fonts
    binary = cv2.adaptiveThreshold(
        filtered, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        25, 
        15
    )

    # 4. Optional: Run a minor morphological closing operation to heal broken kanji brush strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 5. Convert back to standard PIL Image for downstream Tesseract consumption
    return PILImage.fromarray(cleaned)