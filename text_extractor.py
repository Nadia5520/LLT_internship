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
- region_has_significant_text(img, min_words=3, text_area_ratio_thresh=0.02):
    quick helper returning True if region likely contains readable text

Notes:
- Requires pytesseract (pip) and system tesseract binary for OCR.
- If pytesseract is not installed, analyze_text_from_image will raise ImportError.
"""
from typing import Dict, Any
import numpy as np
import cv2

try:
    import pytesseract
    from pytesseract import Output
except Exception:
    pytesseract = None
    Output = None


def _safe_conf_to_int(conf_val):
    """Convert pytesseract conf value (string or number) to int, fallback -1."""
    try:
        # pytesseract sometimes returns strings like "-1" or "96"
        # or numeric floats. Use float first then int.
        if isinstance(conf_val, str):
            conf_val = conf_val.strip()
            if conf_val == "":
                return -1
        conf_f = float(conf_val)
        return int(round(conf_f))
    except Exception:
        return -1


def analyze_text_from_image(img: np.ndarray, preprocess: str = "thresh") -> Dict[str, Any]:
    """
    Perform OCR on a BGR image (numpy array).

    Args:
        img: BGR image (numpy array)
        preprocess: "thresh" (adaptive threshold), "gray", or "none" - preprocessing mode

    Returns:
        dict with keys: text, chars, words, text_area_ratio, boxes
    Raises:
        ImportError if pytesseract is not installed.
    """
    if pytesseract is None or Output is None:
        raise ImportError("pytesseract is required for analyze_text_from_image but is not installed")

    # Convert to RGB for pytesseract
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    if preprocess == "thresh":
        # adaptive threshold can help OCR on overlays and varied lighting
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 21, 10)
        ocr_input = th
    elif preprocess == "gray":
        ocr_input = gray
    else:
        ocr_input = img_rgb

    data = pytesseract.image_to_data(ocr_input, output_type=Output.DICT)
    n_boxes = len(data.get('level', []))
    boxes = []
    h, w = img.shape[:2]
    total_text_area = 0
    text_content = []

    for i in range(n_boxes):
        conf_val = data.get('conf', [])[i] if 'conf' in data else -1
        conf = _safe_conf_to_int(conf_val)

        txt = data.get('text', [])[i].strip() if 'text' in data else ""
        # Filter empty text and low confidence
        if txt == "" or conf < 30:
            continue

        x = int(data.get('left', [])[i])
        y = int(data.get('top', [])[i])
        bw = int(data.get('width', [])[i])
        bh = int(data.get('height', [])[i])

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
    text_area_ratio = float(total_text_area) / float(max(1, w * h))

    return {
        "text": full_text,
        "chars": chars,
        "words": words,
        "text_area_ratio": text_area_ratio,
        "boxes": boxes
    }


def region_has_significant_text(img: np.ndarray, min_words: int = 3, text_area_ratio_thresh: float = 0.02) -> bool:
    """
    Quick helper that runs OCR and decides if the region contains significant readable text.

    Args:
        img: BGR image region
        min_words: minimum number of words to consider "text-heavy"
        text_area_ratio_thresh: minimum fraction of area covered by OCR boxes

    Returns:
        True if region likely contains readable text; False otherwise.
    """
    try:
        res = analyze_text_from_image(img, preprocess="thresh")
    except ImportError:
        # No pytesseract available: return False (can't claim text)
        return False
    except Exception:
        # If OCR failed, be conservative and say there's no significant text
        return False

    if res["words"] >= min_words:
        return True
    if res["text_area_ratio"] >= text_area_ratio_thresh:
        return True
    return False