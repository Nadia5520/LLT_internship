# text_extractor.py
"""
Simple OCR helper utilities using pytesseract.

Provides:
- analyze_text_from_image(img): returns dict with:
    - text: full OCR text
    - chars: number of characters detected
    - words: number of words detected
    - text_area_ratio: fraction of pixels inside OCR boxes (approx) relative to image area
    - boxes: list of detected word boxes
Notes:
- Requires pytesseract (pip) and system tesseract binary.
- If pytesseract is not installed, functions will raise ImportError.
"""
import numpy as np
import cv2

try:
    import pytesseract
    from pytesseract import Output
except Exception as e:
    pytesseract = None
    Output = None

def analyze_text_from_image(img):
    """
    img: BGR image (numpy array)
    returns: dict with text metrics
    """
    if pytesseract is None:
        raise ImportError("pytesseract is required for analyze_text_from_image but is not installed")

    # Convert to RGB for pytesseract
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Optionally pre-process for better OCR
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # adaptive threshold can help OCR on overlays
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 21, 10)

    data = pytesseract.image_to_data(th, output_type=Output.DICT)
    n_boxes = len(data['level'])
    boxes = []
    h, w = img.shape[:2]
    total_text_area = 0
    text_content = []
    for i in range(n_boxes):
                # Get confidence value
        conf_val = data['conf'][i]
        
        # Convert to integer safely
        if isinstance(conf_val, str):
            # If it's a string, check if it's numeric
            conf = int(conf_val) if conf_val.lstrip('-').isdigit() else -1
        else:
            # If it's already a number (int/float), convert directly
            conf = int(conf_val) if conf_val != '' else -1
        txt = data['text'][i].strip()
        if txt == "" or conf < 20:
            continue
        x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        boxes.append({
            "text": txt,
            "conf": conf,
            "box": (x, y, bw, bh)
        })
        total_text_area += bw * bh
        text_content.append(txt)

    full_text = " ".join(text_content)
    chars = len(full_text)
    words = len(full_text.split())
    text_area_ratio = float(total_text_area) / float(max(1, w*h))

    return {
        "text": full_text,
        "chars": chars,
        "words": words,
        "text_area_ratio": text_area_ratio,
        "boxes": boxes
    }