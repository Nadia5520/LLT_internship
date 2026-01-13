#!/usr/bin/env python3
"""
clip_scan_tester.py (open-clip backend)

This version uses open_clip (no TensorFlow) to compute CLIP embeddings and
text-image similarity. It supports:
 - contour-based candidate generation
 - visual prefilters (texture / area)
 - OCR & logo template filtering (optional)
 - optional building/loading of a CLIP embedding index (nearest-neighbor)

Install requirements (in your venv):
  pip install torch torchvision open-clip-torch pillow opencv-python numpy scikit-learn pytesseract

Usage examples:
  # Build index from reference images:
  python clip_scan_tester.py --build-index refs/veillon_images --index-out models/veillon_index.pkl

  # Run scan using index + NN threshold:
  python clip_scan_tester.py --scans scans --out clip_results --index models/veillon_index.pkl --nn-thresh 0.34 --debug

  # Run zero-shot only (no index):
  python clip_scan_tester.py --scans scans --out clip_results --threshold 0.25 --delta 0.08 --debug

Notes:
 - For OCR filtering enable pytesseract and install Tesseract binary.
 - Put small logos/icons you want ignored into the "logos/" folder.
"""
import os
import cv2
import numpy as np
import argparse
import json
import pickle
from pathlib import Path
from PIL import Image

# Optional OCR
try:
    import pytesseract
    from pytesseract import Output as PT_Output
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    PT_Output = None
    OCR_AVAILABLE = False

# open-clip (avoids importing transformers / TF)
try:
    import torch
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use ViT-B-32 to match openai/clip-vit-base-patch32 behavior
    _model, _, _preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    _tokenizer = open_clip.get_tokenizer('ViT-B-32')
    _model.to(device).eval()
except Exception as e:
    raise RuntimeError("open-clip (open_clip) and torch are required. Install with: pip install open-clip-torch torch") from e

# Nearest neighbor support
try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -------------------------
# Helpers
# -------------------------
def candidate_metrics(roi):
    if roi is None or roi.size == 0:
        return {}
    h, w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    std_dev = float(np.std(gray))
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / float(max(1, w * h))
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue_std = float(np.std(hsv[:, :, 0]))
    avg_color = [float(x) for x in np.mean(roi, axis=(0,1)).tolist()]
    return {"w": int(w), "h": int(h), "area": int(w*h), "std": std_dev, "edge_density": edge_density, "hue_std": hue_std, "avg_color_bgr": avg_color}

def is_text_heavy_by_ocr(roi, min_words=3, area_ratio_thresh=0.02, conf_thresh=30):
    if not OCR_AVAILABLE:
        return False
    try:
        img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
        data = pytesseract.image_to_data(th, output_type=PT_Output)
        words = 0
        tot_area = 0
        h, w = roi.shape[:2]
        for i, txt in enumerate(data.get("text", [])):
            if not str(txt).strip():
                continue
            conf = data.get("conf", [])[i]
            try:
                conf_i = int(float(conf))
            except Exception:
                conf_i = -1
            if conf_i < conf_thresh:
                continue
            words += 1
            bw = int(data.get("width", [])[i] or 0)
            bh = int(data.get("height", [])[i] or 0)
            tot_area += bw * bh
        if words >= min_words:
            return True
        if float(tot_area) / float(max(1, w*h)) >= area_ratio_thresh:
            return True
        return False
    except Exception:
        return False

def is_known_logo(roi, templates_folder="logos", match_thresh=0.80):
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
            if th > h_roi or tw > w_roi:
                scale = min(h_roi / th, w_roi / tw)
                if scale <= 0:
                    continue
                tpl = cv2.resize(tpl, (max(1, int(tw*scale)), max(1, int(th*scale))))
            res = cv2.matchTemplate(gray_roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val >= match_thresh:
                return True
    except Exception:
        pass
    return False

def preprocess_roi_for_clip(roi, target=224, trim_border=True, pad=0.02):
    """Trim borders and center-crop ROI, return PIL image for open_clip preprocess."""
    if roi is None or roi.size == 0:
        return None
    img = roi.copy()
    h, w = img.shape[:2]
    if trim_border:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Otsu threshold to separate background
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask = gray != th
        if np.sum(mask) > 0:
            ys, xs = np.where(mask)
            y0, y1 = max(0, ys.min()-2), min(h, ys.max()+2)
            x0, x1 = max(0, xs.min()-2), min(w, xs.max()+2)
            img = img[y0:y1, x0:x1]
    # center-crop largest square then pad white border
    h, w = img.shape[:2]
    min_side = min(h, w) if (h>0 and w>0) else 0
    if min_side > 0:
        cy, cx = h//2, w//2
        half = min_side//2
        img = img[cy-half:cy-half+min_side, cx-half:cx-half+min_side]
    if pad > 0 and img.size != 0:
        ph = int(pad * img.shape[0])
        pw = int(pad * img.shape[1])
        img = cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_CONSTANT, value=[255,255,255])
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return pil

def clip_text_image_similarity_preprocessed(roi, prompts):
    """Preprocess ROI and compute open-clip similarity scores against prompts."""
    pil = preprocess_roi_for_clip(roi)
    if pil is None:
        return np.zeros(len(prompts), dtype=float)
    img_t = _preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feat = _model.encode_image(img_t)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        tokens = _tokenizer(prompts)
        tokens = tokens.to(device)
        text_feat = _model.encode_text(tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        sims = (image_feat @ text_feat.T).squeeze(0).cpu().numpy()
    return sims

# Candidate proposer (contours)
def find_candidate_regions(img, morph_kernel=5, canny1=50, canny2=150, min_area_px=2000, min_area_frac=0.001, max_area_frac=0.9):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, canny1, canny2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    image_area = w*h
    for cnt in contours:
        area = cv2.contourArea(cnt)
        min_area = max(min_area_px, int(min_area_frac*image_area))
        if area < min_area or area > max_area_frac*image_area:
            continue
        x,y,rw,rh = cv2.boundingRect(cnt)
        pad = int(0.01 * max(rw, rh))
        x0 = max(0, x-pad); y0 = max(0, y-pad)
        x1 = min(w, x+rw+pad); y1 = min(h, y+rh+pad)
        regions.append((x0, y0, x1-x0, y1-y0))
    # merge overlaps
    merged = []
    for r in regions:
        x,y,rw,rh = r
        merged_flag = False
        for i,(mx,my,mw,mh) in enumerate(merged):
            ix = max(x,mx); iy = max(y,my); ax = min(x+rw, mx+mw); ay = min(y+rh, my+mh)
            if ix < ax and iy < ay:
                nx = min(x,mx); ny = min(y,my); nx2 = max(x+rw, mx+mw); ny2 = max(y+rh, my+mh)
                merged[i] = (nx, ny, nx2-nx, ny2-ny)
                merged_flag = True
                break
        if not merged_flag:
            merged.append(r)
    return merged

# Index building and querying (open-clip embeddings + sklearn NN)
def build_index(ref_folder, out_path='veillon_index.pkl'):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required to build index (pip install scikit-learn).")
    files = [f for f in os.listdir(ref_folder) if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))]
    embeddings = []
    fnames = []
    for f in files:
        p = os.path.join(ref_folder, f)
        img = cv2.imread(p)
        if img is None:
            continue
        pil = preprocess_roi_for_clip(img)
        if pil is None:
            continue
        img_t = _preprocess(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = _model.encode_image(img_t)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.squeeze(0).cpu().numpy().astype(np.float32)
        embeddings.append(emb)
        fnames.append(f)
    if not embeddings:
        raise RuntimeError("No embeddings created; check reference folder.")
    emb_mat = np.vstack(embeddings)
    nn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(emb_mat)
    with open(out_path, 'wb') as fh:
        pickle.dump({'filenames': fnames, 'embeddings': emb_mat}, fh)
    print("Saved index to", out_path)
    return out_path

def load_index(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)
    d = pickle.load(open(index_path,'rb'))
    fnames = d['filenames']
    emb = d['embeddings']
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required for NN queries.")
    nn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(emb)
    return {'nn': nn, 'filenames': fnames, 'embeddings': emb}

def query_index(index_obj, roi, top_k=3):
    pil = preprocess_roi_for_clip(roi)
    img_t = _preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = _model.encode_image(img_t)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.squeeze(0).cpu().numpy().astype(np.float32)
    dists, idxs = index_obj['nn'].kneighbors(emb.reshape(1,-1), n_neighbors=top_k, return_distance=True)
    results = []
    for dist, idx in zip(dists[0], idxs[0]):
        results.append({'filename': index_obj['filenames'][int(idx)], 'dist': float(dist)})
    return results

# Prompts
def pretty_prompts():
    return [
        "a painting by Margo Veillon",
        "a Margo Veillon artwork",
        "an oil painting by Margo Veillon",
        "a watercolor painting by Margo Veillon",
        "a painting",
        "an artwork",
        "a photograph",
        "a logo",
        "a screenshot",
        "an icon"
    ]

# Main runner
def run_on_scans(scans_folder, out_folder, threshold=0.25, delta=0.08, nn_thresh=0.32,
                 save_crops=True, visualize=True, logos_folder="logos", debug=False, index_obj=None,
                 prefilter_std=12.0, prefilter_min_area=1500):
    os.makedirs(out_folder, exist_ok=True)
    crops_folder = os.path.join(out_folder, "crops")
    vis_folder = os.path.join(out_folder, "visualizations")
    skipped_folder = os.path.join(out_folder, "skipped")
    os.makedirs(crops_folder, exist_ok=True)
    os.makedirs(vis_folder, exist_ok=True)
    os.makedirs(skipped_folder, exist_ok=True)
    os.makedirs(logos_folder, exist_ok=True)

    scans = [f for f in os.listdir(scans_folder) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
    prompts = pretty_prompts()
    results = {}

    for scan in scans:
        img_path = os.path.join(scans_folder, scan)
        img = cv2.imread(img_path)
        if img is None:
            print("Could not read:", img_path); continue
        vis = img.copy()
        candidates = find_candidate_regions(img)
        print(f"[{scan}] {len(candidates)} candidates found")
        image_results = []
        for i, (x,y,rw,rh) in enumerate(candidates):
            roi = img[y:y+rh, x:x+rw]
            met = candidate_metrics(roi)
            # prefilter by texture + area to ditch text-like regions
            if met.get('area',0) < prefilter_min_area or met.get('std',0) < prefilter_std:
                if debug:
                    cv2.imwrite(os.path.join(skipped_folder, f"{Path(scan).stem}_cand{i+1}_prefail.jpg"), roi)
                continue
            # logos
            if is_known_logo(roi, templates_folder=logos_folder):
                if debug:
                    cv2.imwrite(os.path.join(skipped_folder, f"{Path(scan).stem}_cand{i+1}_logo.jpg"), roi)
                continue
            # OCR
            if is_text_heavy_by_ocr(roi):
                if debug:
                    cv2.imwrite(os.path.join(skipped_folder, f"{Path(scan).stem}_cand{i+1}_text.jpg"), roi)
                continue
            # CLIP zero-shot (open-clip)
            try:
                sims = clip_text_image_similarity_preprocessed(roi, prompts)
            except Exception as e:
                print("CLIP candidate error:", e)
                if debug:
                    cv2.imwrite(os.path.join(skipped_folder, f"{Path(scan).stem}_cand{i+1}_clipfail.jpg"), roi)
                continue
            mv_score = float(sims[0])
            other_max = float(max(sims[1:])) if len(sims) > 1 else 0.0
            decision = (mv_score - other_max >= delta) and (mv_score >= threshold)
            entry = {"candidate": i+1, "coords":[int(x),int(y),int(rw),int(rh)], "mv_score":mv_score, "other_max":other_max, "decision": bool(decision)}
            # NN index override / additional check
            if index_obj is not None:
                try:
                    matches = query_index(index_obj, roi, top_k=3)
                    best = matches[0]
                    sim = 1.0 - best['dist']  # convert cosine distance -> similarity
                    entry['nn_best_sim'] = sim
                    entry['nn_matches'] = matches
                    if sim >= nn_thresh:
                        decision = True
                        entry['decision'] = True
                except Exception as e:
                    if debug:
                        print("NN query failed:", e)
            image_results.append(entry)

            label = f"MV:{mv_score:.2f}"
            color = (0,255,255) if entry['decision'] else (0,0,255)
            cv2.rectangle(vis, (x,y), (x+rw, y+rh), color, 2)
            cv2.putText(vis, label, (x, max(10,y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if entry['decision'] and save_crops:
                fname = f"{Path(scan).stem}_cand{i+1}_mv{mv_score:.2f}.jpg"
                cv2.imwrite(os.path.join(crops_folder, fname), roi)
            else:
                if debug:
                    cv2.imwrite(os.path.join(skipped_folder, f"{Path(scan).stem}_cand{i+1}_rej_mv{mv_score:.2f}.jpg"), roi)

        if visualize:
            cv2.imwrite(os.path.join(vis_folder, f"vis_{scan}"), vis)
        accepted = sum(1 for e in image_results if e['decision'])
        print(f"  Accepted: {accepted}/{len(image_results)}")
        results[scan] = image_results

    report = os.path.join(out_folder, "clip_scan_report.json")
    with open(report, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2)
    print("Report saved to", report)
    print("Crops:", os.path.join(out_folder, "crops"))
    print("Visualizations:", os.path.join(out_folder, "visualizations"))
    print("Skipped:", os.path.join(out_folder, "skipped"))

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scans", default="scans", help="Input folder")
    p.add_argument("--out", default="clip_results", help="Output folder")
    p.add_argument("--threshold", type=float, default=0.25, help="Min mv_score to accept")
    p.add_argument("--delta", type=float, default=0.08, help="mv_score - max(other) >= delta")
    p.add_argument("--nn-thresh", type=float, default=0.32, help="If using index: min NN sim to accept")
    p.add_argument("--build-index", default=None, help="Path to reference images folder to build index")
    p.add_argument("--index-out", default="veillon_index.pkl", help="Where to save built index")
    p.add_argument("--index", default=None, help="Path to prebuilt index .pkl to use for NN lookup")
    p.add_argument("--logos-folder", default="logos", help="Folder with small logo templates to ignore")
    p.add_argument("--no-visualize", action="store_true", help="Don't write visualization images")
    p.add_argument("--no-save-crops", action="store_true", help="Don't save accepted crop images")
    p.add_argument("--debug", action="store_true", help="Save rejected candidates and print debug info")
    p.add_argument("--prefilter-std", type=float, default=12.0, help="Std-dev prefilter to remove text/flat regions")
    p.add_argument("--prefilter-min-area", type=int, default=1500, help="Min area prefilter")
    args = p.parse_args()

    index_obj = None
    if args.build_index:
        print("Building index from", args.build_index)
        idxp = build_index(args.build_index, out_path=args.index_out)
        print("Built index:", idxp)
    if args.index:
        print("Loading index:", args.index)
        index_obj = load_index(args.index)

    run_on_scans(args.scans, args.out, threshold=args.threshold, delta=args.delta, nn_thresh=args.nn_thresh,
                 save_crops=not args.no_save_crops, visualize=not args.no_visualize, logos_folder=args.logos_folder,
                 debug=args.debug, index_obj=index_obj, prefilter_std=args.prefilter_std, prefilter_min_area=args.prefilter_min_area)