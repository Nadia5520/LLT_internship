# scan_pipeline.py
"""
Scan -> Margo Veillon artwork extractor pipeline

This script builds on the previous extractor and adds:
- OCR-based filtering (pytesseract) to remove text/UI-heavy crops (e.g. Google Translate overlays)
- A classifier step to keep only crops predicted as Margo Veillon (if a model is provided)
- Options to save non-Margo crops for inspection

Usage:
    python scan_pipeline.py --input scans --output extracted_artworks --viz visualizations \
        --model margo_veillon_classifier.keras --require-classifier --clean

Dependencies:
    pip install opencv-python-headless numpy tensorflow pytesseract Pillow

System dependency:
    - Tesseract OCR: required for OCR filtering.
      On Ubuntu: apt-get install tesseract-ocr
      On macOS: brew install tesseract
"""
import os
import uuid
from pathlib import Path
import argparse

import cv2
import numpy as np

# Optional: tensorflow and pytesseract are imported lazily to allow running without them
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception:
    tf = None
    load_model = None

try:
    import pytesseract
except Exception:
    pytesseract = None

# Local helper module (provided below as text_extractor.py)
from text_extractor import analyze_text_from_image

# ---- utility functions (resize, transform) ----
def resize_max(image, max_dim=1600):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image, 1.0
    scale = max_dim / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def order_quad(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(img, pts):
    rect = order_quad(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    if maxWidth <= 0 or maxHeight <= 0:
        return None
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

# ---- detection pipeline (same approach as before, kept simple) ----
def detect_artworks(img, min_area_rel=0.004, max_area_rel=0.95):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    th = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(denoised, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    combined = cv2.bitwise_or(closed, edges)

    contours_info = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_rel = area / float(w * h)
        if area_rel < min_area_rel or area_rel > max_area_rel:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, bw, bh = cv2.boundingRect(approx)
        aspect = bw / float(bh) if bh > 0 else 0
        if aspect < 0.3 or aspect > 3.5:
            continue
        if bw < 60 or bh < 60:
            continue
        roi = img[y:y+bh, x:x+bw]
        if roi.size == 0:
            continue
        # Color variety check
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hue_std = float(np.std(hsv[:, :, 0]))
        except Exception:
            hue_std = 0.0
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges_roi = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges_roi > 0) / float(bw * bh)
        if hue_std < 8 and edge_density < 0.02:
            continue
        if edge_density < 0.01:
            continue

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            candidates.append({
                "type": "quad",
                "pts": pts,
                "bbox": (x, y, bw, bh),
                "score": area_rel
            })
        else:
            candidates.append({
                "type": "rect",
                "bbox": (x, y, bw, bh),
                "score": area_rel
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates

# ---- processing per file ----
def process_file(path_in: Path, out_dir: Path, viz_dir: Path, model, args):
    img_orig = cv2.imread(str(path_in))
    if img_orig is None:
        print(f"  ❌ Could not read {path_in.name}")
        return 0
    resized, scale = resize_max(img_orig, max_dim=args.max_dim)
    scale_inv = 1.0 / scale
    candidates = detect_artworks(resized,
                                 min_area_rel=args.min_area_rel,
                                 max_area_rel=args.max_area_rel)
    print(f"  Size: {img_orig.shape[1]}x{img_orig.shape[0]} (scaled to {resized.shape[1]}x{resized.shape[0]})")
    print(f"  Detected candidates: {len(candidates)}")

    vis = resized.copy()
    extracted_count = 0

    for i, c in enumerate(candidates):
        x, y, w, h = c["bbox"]
        # obtain crop at original resolution
        if c["type"] == "quad":
            # Try to get a good crop on the original image if scaling was used
            try:
                pts_orig = (c["pts"] * scale_inv).astype(np.float32)
                crop = four_point_transform(img_orig, pts_orig)
                if crop is None:
                    # fallback to scaled warp
                    crop = four_point_transform(resized, c["pts"])
            except Exception:
                crop = four_point_transform(resized, c["pts"])
        else:
            x_o = int(round(x * scale_inv))
            y_o = int(round(y * scale_inv))
            w_o = int(round(w * scale_inv))
            h_o = int(round(h * scale_inv))
            crop = img_orig[y_o:y_o+h_o, x_o:x_o+w_o]

        if crop is None or crop.size == 0:
            continue
        ch, cw = crop.shape[:2]
        if ch < 40 or cw < 40:
            continue

        # 1) OCR-based text filtering
        text_info = None
        if args.ocr and pytesseract is not None:
            text_info = analyze_text_from_image(crop)
            # if overlay-like (e.g., big text area or "Translate" word), skip
            if text_info["text_area_ratio"] >= args.max_text_area_ratio:
                print(f"    ❌ Rejected candidate #{i+1} due to large text area ({text_info['text_area_ratio']:.2f})")
                continue
            if text_info["chars"] >= args.max_text_chars:
                print(f"    ❌ Rejected candidate #{i+1} due to many chars ({text_info['chars']})")
                continue
            # Google Translate heuristic
            if "translate" in text_info["text"].lower() or "google" in text_info["text"].lower():
                print(f"    ❌ Rejected candidate #{i+1} due to detected UI text ('translate'/'google')")
                continue
        elif args.ocr and pytesseract is None:
            print("    ⚠️ OCR requested but pytesseract not installed; skipping OCR filtering")

        # 2) classifier filtering (if model provided)
        is_margo = None
        score = None
        if model is not None:
            try:
                # preprocess crop to model input size
                inp = cv2.resize(crop, (args.class_img_w, args.class_img_h))
                inp = inp.astype("float32") / 255.0
                inp = np.expand_dims(inp, axis=0)
                pred = model.predict(inp, verbose=0)
                # model trained with sigmoid single-output => pred is [[prob]]
                score = float(pred.squeeze())
                is_margo = score >= args.classifier_threshold
            except Exception as e:
                print(f"    ⚠️ Classification error: {e}")
                is_margo = None

            if is_margo is False:
                if args.save_non_margo:
                    sub = out_dir / "non_margo"
                    sub.mkdir(parents=True, exist_ok=True)
                    out_name = f"{path_in.stem}_art_{i+1}_nonmargo_{uuid.uuid4().hex[:8]}.jpg"
                    cv2.imwrite(str(sub / out_name), crop, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                    print(f"    ⓘ Saved non-Margo candidate to '{sub.name}/{out_name}' (score={score:.3f})")
                else:
                    print(f"    ❌ Rejected candidate #{i+1} by classifier (score={score:.3f})")
                continue  # drop non-Margo
            else:
                print(f"    ✅ Classifier accepted candidate #{i+1} (score={score:.3f})")

        elif args.require_classifier:
            print("    ❌ No classifier available but --require-classifier specified => skipping candidate")
            continue

        # Save accepted crop
        base = path_in.stem
        unique = uuid.uuid4().hex[:8]
        out_name = f"{base}_margo_{i+1}_{unique}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        extracted_count += 1

        # Visualization
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
        label = f"#{i+1}"
        if score is not None:
            label += f" {score:.2f}"
        cv2.putText(vis, label, (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # save visualization
    vis_name = f"vis_{path_in.stem}.jpg"
    cv2.imwrite(str(viz_dir / vis_name), vis)
    return extracted_count

# ---- CLI and main ----
def parse_args():
    p = argparse.ArgumentParser(description="Scan -> Margo Veillon extractor with OCR + classifier filtering")
    p.add_argument("--input", "-i", default="scans", help="Input folder")
    p.add_argument("--output", "-o", default="extracted_artworks", help="Output folder for accepted crops (Margo)")
    p.add_argument("--viz", "-v", default="visualizations", help="Visualizations folder")
    p.add_argument("--clean", action="store_true", help="Clean output and viz folders before running")
    p.add_argument("--max-dim", type=int, default=1600, help="Max processing dimension")
    p.add_argument("--min-area-rel", type=float, default=0.004, help="Min candidate area rel")
    p.add_argument("--max-area-rel", type=float, default=0.95, help="Max candidate area rel")
    p.add_argument("--jpeg-quality", type=int, default=95, help="Output JPEG quality")
    p.add_argument("--ocr", action="store_true", help="Enable OCR-based text filtering (requires pytesseract + tesseract)")
    p.add_argument("--max-text-area-ratio", type=float, default=0.30,
                   help="Reject candidate when detected text area occupies more than this fraction of crop")
    p.add_argument("--max-text-chars", type=int, default=60, help="Reject if OCR returns this many or more characters")
    p.add_argument("--model", type=str, default="margo_veillon_classifier.keras", help="Path to classifier model (keras .keras or .h5)")
    p.add_argument("--require-classifier", action="store_true",
                   help="If set, candidates are only saved when classifier is available and accepts them")
    p.add_argument("--classifier-threshold", type=float, default=0.5, help="Sigmoid threshold to consider Margo")
    p.add_argument("--class-img-w", type=int, default=224, help="Classifier input width")
    p.add_argument("--class-img-h", type=int, default=224, help="Classifier input height")
    p.add_argument("--save-non-margo", action="store_true", help="Save rejected non-Margo crops to output/non_margo for inspection")
    return p.parse_args()

def mkdir_fresh(path: Path, clean: bool = False):
    if path.exists():
        if clean:
            for p in path.iterdir():
                try:
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        for sub in p.rglob('*'):
                            if sub.is_file():
                                sub.unlink()
                        p.rmdir()
                except Exception:
                    pass
    else:
        path.mkdir(parents=True, exist_ok=True)

def main():
    args = parse_args()
    in_folder = Path(args.input)
    out_folder = Path(args.output)
    viz_folder = Path(args.viz)

    if not in_folder.exists() or not in_folder.is_dir():
        print(f"❌ Input folder '{in_folder}' does not exist or is not a directory.")
        return

    mkdir_fresh(out_folder, clean=args.clean)
    mkdir_fresh(viz_folder, clean=args.clean)

    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    files = [p for p in sorted(in_folder.iterdir()) if p.suffix.lower() in exts]
    if not files:
        print("❌ No image files found in input folder.")
        return

    # Load classifier if requested/available
    model = None
    if args.model and load_model is not None and Path(args.model).exists():
        try:
            model = load_model(args.model)
            print(f"Loaded classifier model from '{args.model}'")
        except Exception as e:
            print(f"⚠️ Could not load model '{args.model}': {e}")
            model = None
    else:
        if args.model:
            print(f"⚠️ Model file '{args.model}' not found or tensorflow not available. classifier disabled.")

    print("="*60)
    print("ARTWORK EXTRACTOR - RUN")
    print("="*60)
    print(f"Found {len(files)} files to process")

    total = 0
    for f in files:
        print(f"\n🔍 Processing: {f.name}")
        try:
            count = process_file(f, out_folder, viz_folder, model, args)
            print(f"  Extracted accepted (Margo) crops: {count}")
            total += count
        except Exception as e:
            print(f"  ❌ Error processing {f.name}: {e}")

    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY:")
    print(f"   Files processed: {len(files)}")
    print(f"   Artworks extracted (accepted as Margo): {total}")
    print(f"   Clean artworks: '{out_folder}/'")
    print(f"   Visualizations: '{viz_folder}/'")
    print("="*60)

if __name__ == "__main__":
    main()