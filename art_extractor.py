#!/usr/bin/env python3
"""
art_extractor.py (debuggable / tunable)

Improved, parameterized extractor with a --debug mode that saves ALL candidate regions
before filtering and prints diagnostic metrics for each candidate. Use this to
inspect why some artworks are not being detected and to adjust thresholds.

Usage examples:
  # quick run with defaults (heuristics + optional OCR)
  python art_extractor.py --scans scans

  # debug mode: save all pre-filter candidates to debug_candidates/
  python art_extractor.py --scans scans --debug --save-candidates

  # relax size thresholds if artworks are small in your scans
  python art_extractor.py --min-area-px 4000 --min-area-frac 0.003 --min-width 60 --min-height 60 --debug --save-candidates

  # use classifier if you have a trained model and classifier_inference.py
  python art_extractor.py --model models/margo_classifier.keras --threshold 0.6

Outputs:
 - extracted artwork crops in OUTPUT_FOLDER (default: extracted_artworks_clean)
 - visualizations in VISUALIZATION_FOLDER
 - debug_candidates/ when --debug is used (every candidate prior to heuristic filtering)
"""
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import time
import json

# Optional integrations
try:
    from text_extractor import region_has_significant_text, analyze_text_from_image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

Classifier = None
try:
    from classifier_inference import Classifier
except Exception:
    Classifier = None

def ensure_folders(*folders):
    for f in folders:
        os.makedirs(f, exist_ok=True)

def candidate_metrics(roi):
    """Compute diagnostics used by heuristics for debugging."""
    if roi is None or roi.size == 0:
        return {}
    h, w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    std_dev = float(np.std(gray))
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / float(max(1, w * h))
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue_std = float(np.std(hsv[:,:,0]))
    avg_color = [float(x) for x in np.mean(roi, axis=(0,1)).tolist()]  # BGR
    return {
        "w": int(w), "h": int(h),
        "area": int(w*h),
        "aspect": float(w)/float(h) if h>0 else 0,
        "std_dev_gray": std_dev,
        "edge_density": edge_density,
        "hue_std": hue_std,
        "avg_color_bgr": avg_color
    }

def is_ui_element(roi, edge_density_thresh, hue_std_thresh, avg_color_thresh):
    if roi is None or roi.size == 0:
        return True
    m = candidate_metrics(roi)
    if m["w"] < 80 or m["h"] < 80:
        return True
    if m["hue_std"] < hue_std_thresh:
        return True
    if m["edge_density"] > edge_density_thresh:
        return True
    avg = m["avg_color_bgr"]
    if (avg[0] > 130 and avg[1] < 120 and avg[2] < 120) or (np.std(avg) < avg_color_thresh and np.mean(avg) > 160):
        return True
    return False

def is_known_logo(roi, templates_folder="logos", match_thresh=0.80):
    """Return True if roi matches any small logo template (template matching)."""
    try:
        if not os.path.isdir(templates_folder):
            return False
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h_roi, w_roi = gray_roi.shape[:2]
        for tfn in os.listdir(templates_folder):
            tp = os.path.join(templates_folder, tfn)
            tpl = cv2.imread(tp, cv2.IMREAD_GRAYSCALE)
            if tpl is None:
                continue
            th, tw = tpl.shape[:2]
            # skip templates bigger than ROI
            if th > h_roi or tw > w_roi:
                # downscale template if ROI smaller (optional)
                scale = min(h_roi / th, w_roi / tw)
                if scale <= 0:
                    continue
                tpl = cv2.resize(tpl, (max(1, int(tw*scale)), max(1, int(th*scale))), interpolation=cv2.INTER_AREA)
            # normalized cross-correlation
            try:
                res = cv2.matchTemplate(gray_roi, tpl, cv2.TM_CCOEFF_NORMED)
            except Exception:
                continue
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val >= match_thresh:
                return True
    except Exception:
        return False
    return False

def is_likely_artwork(roi, image_area,
                      min_area_px, min_area_frac,
                      max_area_frac,
                      std_dev_thresh, edge_density_thresh,
                      hue_std_thresh, min_width, min_height):
    if roi is None or roi.size == 0:
        return False
    h, w = roi.shape[:2]
    area = w*h
    min_area = max(min_area_px, int(min_area_frac * image_area))
    max_area = int(max_area_frac * image_area)
    if area < min_area or area > max_area:
        return False
    if w < min_width or h < min_height:
        return False
    aspect = float(w)/float(h)
    if aspect < 0.25 or aspect > 4.0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < std_dev_thresh:
        return False
    if is_ui_element(roi, edge_density_thresh=edge_density_thresh, hue_std_thresh=hue_std_thresh, avg_color_thresh=10):
        return False
    # OCR: if available, reject text-heavy regions
    if OCR_AVAILABLE:
        try:
            if region_has_significant_text(roi, min_words=3, text_area_ratio_thresh=0.03):
                return False
        except Exception:
            pass
    return True

def find_candidate_regions(img, morph_kernel=(9,9), canny1=30, canny2=100, min_area_px=15000, min_area_frac=0.01, max_area_frac=0.9):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, canny1, canny2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    image_area = w*h
    for cnt in contours:
        area = cv2.contourArea(cnt)
        min_area = max(min_area_px, min_area_frac*image_area)
        if area < min_area or area > max_area_frac*image_area:
            continue
        x, y, rw, rh = cv2.boundingRect(cnt)
        pad = int(0.02 * max(rw, rh))
        x0 = max(0, x-pad)
        y0 = max(0, y-pad)
        x1 = min(w, x+rw+pad)
        y1 = min(h, y+rh+pad)
        regions.append((x0, y0, x1-x0, y1-y0))
    # merge overlaps
    merged = []
    for r in regions:
        x, y, rw, rh = r
        merged_flag = False
        for i, (mx, my, mw, mh) in enumerate(merged):
            ix = max(x, mx); iy = max(y, my)
            ax = min(x+rw, mx+mw); ay = min(y+rh, my+mh)
            if ix < ax and iy < ay:
                nx = min(x, mx); ny = min(y, my)
                nx2 = max(x+rw, mx+mw); ny2 = max(y+rh, my+mh)
                merged[i] = (nx, ny, nx2-nx, ny2-ny)
                merged_flag = True
                break
        if not merged_flag:
            merged.append(r)
    return merged

def annotate_and_save(vis_img, x, y, w, h, label_text, color):
    cv2.rectangle(vis_img, (x,y), (x+w,y+h), color, 3)
    cv2.putText(vis_img, label_text, (x, max(0,y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def process_scans(args):
    ensure_folders(args.scans, args.output, args.vis, args.debug_folder)
    scan_files = [f for f in os.listdir(args.scans) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
    if not scan_files:
        print("No images found in", args.scans)
        return

    clf = None
    if args.model and os.path.exists(args.model) and Classifier is not None:
        try:
            print("Loading classifier:", args.model)
            clf = Classifier(args.model, feature_model_path=args.feature_model)
        except Exception as e:
            print("Failed to load classifier:", e)
            clf = None
    else:
        if args.model:
            print("Model path provided but classifier_inference not available or model missing.")
        else:
            print("No classifier model provided — running heuristics + OCR (if available).")

    total_saved = 0
    t0 = time.time()
    for scan_file in scan_files:
        print(f"\nProcessing: {scan_file}")
        img_path = os.path.join(args.scans, scan_file)
        img = cv2.imread(img_path)
        if img is None:
            print("  Could not read", img_path); continue
        h, w = img.shape[:2]
        vis = img.copy()
        candidates = find_candidate_regions(
            img,
            morph_kernel=(args.morph_kernel, args.morph_kernel),
            canny1=args.canny1, canny2=args.canny2,
            min_area_px=args.min_area_px, min_area_frac=args.min_area_frac, max_area_frac=args.max_area_frac
        )
        print(f"  {len(candidates)} candidate regions found (pre-filter).")

        basename = os.path.splitext(scan_file)[0]
        saved_for_image = 0

        for i, (x, y, rw, rh) in enumerate(candidates):
            roi = img[y:y+rh, x:x+rw]
            metrics = candidate_metrics(roi)
            # if debug: save every candidate before filtering and write diagnostics
            if args.debug:
                debug_name = f"{basename}_cand_{i+1}_w{metrics.get('w',0)}_h{metrics.get('h',0)}_area{metrics.get('area',0)}.jpg"
                debug_path = os.path.join(args.debug_folder, debug_name)
                try:
                    cv2.imwrite(debug_path, roi)
                except Exception:
                    pass
                # write json with metrics
                try:
                    with open(os.path.join(args.debug_folder, debug_name + ".json"), "w", encoding="utf-8") as jf:
                        d = {"coords": (int(x),int(y),int(rw),int(rh)), "metrics": metrics}
                        # OCR diagnostics
                        if OCR_AVAILABLE:
                            try:
                                ocr = analyze_text_from_image(roi, preprocess="thresh")
                                d["ocr"] = {"words": ocr.get("words",0), "chars": ocr.get("chars",0), "text_area_ratio": ocr.get("text_area_ratio",0)}
                            except Exception:
                                d["ocr"] = {"error": "ocr_failed"}
                        json.dump(d, jf, indent=2)
                except Exception:
                    pass
                # print diagnostics to console (short)
                print(f"   Candidate {i+1}: w={metrics.get('w')} h={metrics.get('h')} area={metrics.get('area')} aspect={metrics.get('aspect'):.2f} std={metrics.get('std_dev_gray'):.1f} edge={metrics.get('edge_density'):.3f} hue_std={metrics.get('hue_std'):.2f}")

            # Determine if artwork using heuristics (tunable)
            keep = is_likely_artwork(
                roi, image_area=w*h,
                min_area_px=args.min_area_px, min_area_frac=args.min_area_frac,
                max_area_frac=args.max_area_frac,
                std_dev_thresh=args.std_dev_thresh, edge_density_thresh=args.edge_density_thresh,
                hue_std_thresh=args.hue_std_thresh, min_width=args.min_width, min_height=args.min_height
            )

            # If classifier is present we may further filter (but user can set --save-non-matching)
            label_text = "Artwork"
            color = (0,255,0)
            save_crop = keep or args.save_candidates
            if clf is not None:
                try:
                    prob, label = clf.predict_crop(roi)
                    prob = float(prob)
                    label_text = f"{label} {prob:.2f}"
                    # decide positive if label contains margo/veillon OR binary positive above threshold
                    is_positive = False
                    names = getattr(clf, "class_names", None)
                    if names and len(names) == 2:
                        positive_name = names[1]
                        if label == positive_name and prob >= args.threshold:
                            is_positive = True
                    else:
                        if isinstance(label, str) and ("margo" in label.lower() or "veillon" in label.lower()) and prob >= args.threshold:
                            is_positive = True
                    if is_positive:
                        color = (0,255,255)
                        save_crop = True
                    else:
                        color = (0,0,255)
                        if not args.save_non_matching:
                            save_crop = False
                except Exception as e:
                    print("   Classifier error:", e)

           
            # skip if logo detected
            if is_known_logo(roi, templates_folder="logos", match_thresh=0.80):
                if args.debug:
                    print("   Skipping known logo in candidate", i+1)
                continue

             # Save crop if desired
            if save_crop:
                out_name = f"{basename}_artwork_{i+1}.jpg"
                out_path = os.path.join(args.output, out_name)
                try:
                    cv2.imwrite(out_path, roi); total_saved += 1; saved_for_image += 1
                except Exception as e:
                    print("   Failed to save crop:", e)

            annotate_and_save(vis, x, y, rw, rh, label_text, color)

        # save visualization
        vis_path = os.path.join(args.vis, f"vis_{scan_file}")
        try:
            cv2.imwrite(vis_path, vis)
        except Exception:
            pass

        print(f"  Saved {saved_for_image} crops for image.")

    t1 = time.time()
    print("\n" + "="*50)
    print("Done. Total saved crops:", total_saved)
    print("Debug folder:", args.debug_folder if args.debug else "(not used)")
    print("Vis folder:", args.vis)
    print("Elapsed: %.1fs" % (t1-t0))
    print("="*50)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scans", default="scans", help="Input folder")
    p.add_argument("--output", default="extracted_artworks_clean", help="Where to save final crops")
    p.add_argument("--vis", default="detection_visualizations", help="Where to save visualizations")
    p.add_argument("--debug", action="store_true", help="Save all pre-filter candidates and diagnostics")
    p.add_argument("--debug-folder", dest="debug_folder", default="debug_candidates", help="Where to save debug candidates")
    p.add_argument("--model", default=None, help="Path to classifier model (.keras)")
    p.add_argument("--feature_model", default=None, help="Path to feature extractor model (.keras)")
    p.add_argument("--threshold", default=0.6, type=float, help="Classifier acceptance threshold")
    p.add_argument("--save-candidates", action="store_true", help="Save all pre-filter candidates (useful for debugging)")
    p.add_argument("--save-non-matching", action="store_true", help="Keep classifier non-matching crops")
    # tuning params
    p.add_argument("--min-area-px", type=int, default=15000, help="Minimum absolute area for candidate")
    p.add_argument("--min-area-frac", type=float, default=0.01, help="Minimum fraction of image area")
    p.add_argument("--max-area-frac", type=float, default=0.9, help="Maximum fraction of image area")
    p.add_argument("--std-dev-thresh", dest="std_dev_thresh", type=float, default=12.0, help="Stddev threshold (texture)")
    p.add_argument("--edge-density-thresh", dest="edge_density_thresh", type=float, default=0.35, help="Edge density threshold to mark UI")
    p.add_argument("--hue-std-thresh", dest="hue_std_thresh", type=float, default=5.0, help="Hue stddev threshold to mark flat-color UI")
    p.add_argument("--min-width", type=int, default=150, help="Min width for artwork")
    p.add_argument("--min-height", type=int, default=150, help="Min height for artwork")
    p.add_argument("--morph-kernel", type=int, default=9, help="Size of morphological kernel for contour cleanup")
    p.add_argument("--canny1", type=int, default=30, help="Canny param1")
    p.add_argument("--canny2", type=int, default=100, help="Canny param2")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_scans(args)