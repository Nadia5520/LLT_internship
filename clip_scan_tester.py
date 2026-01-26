#!/usr/bin/env python3
"""
clip_scan_tester.py (open-clip backend) - Prompt/decision fixes to improve recall
for non-training examples of the same artist while still rejecting other artists.

What changed (high level):
 - Better prompt engineering: grouped prompts (margo, generic-art, negative artists).
 - Decision rule uses artworkness (is-this-a-painting) + relative margo-vs-others
   comparisons and explicit negative-artist checks (e.g. Van Gogh).
 - More robust use of CLIP model logit scale -> softmax probabilities.
 - Added configurable thresholds for: artworkness, margoness, negative-artist gap,
   and probability threshold. Defaults tuned to increase recall for unseen Margo
   examples while rejecting other artists.
 - Kept index-based NN confirmation (if you build / load an index) as a strong
   positive signal (overrides zero-shot only when present).

Usage:
  # Run scans with default thresholds (tweak with CLI if needed)
  python clip_scan_tester.py --scans scans --out clip_results --debug

  # Build/load an index and require NN confirmation:
  python clip_scan_tester.py --build-index refs/veillon_images --index-out models/veillon_enhanced.pkl
  python clip_scan_tester.py --scans scans --index models/veillon_enhanced.pkl --debug
"""
import os
import cv2
import numpy as np
import argparse
import json
import pickle
from pathlib import Path
from PIL import Image
import sys

# Optional OCR
try:
    import pytesseract
    from pytesseract import Output as PT_Output
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    PT_Output = None
    OCR_AVAILABLE = False

# open-clip
try:
    import torch
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model, _, _preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    _tokenizer = open_clip.get_tokenizer('ViT-B-16')
    _model.to(device).eval()
    print(f"Loaded CLIP model: ViT-B-16 on {device}")
except Exception as e:
    raise RuntimeError("open-clip and torch are required. Install with: pip install open-clip-torch torch") from e

# Nearest neighbor support (optional)
try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -------------------------
# Prompt groups (improved)
# -------------------------
def grouped_prompts():
    """
    Returns dict with prompt groups:
      - 'margo': many variations for Margo Veillon (to increase zero-shot recall)
      - 'art': generic artwork prompts (to check artworkness)
      - 'other': other common labels (logo, photo) to penalize non-art
      - 'negatives': explicit negative artist prompts (Vincent van Gogh etc.)
      - 'all_prompts': flattened list (used for scoring)
    """
    margo = [
        "a painting by Margo Veillon",
        "a Margo Veillon artwork",
        "an artwork by Margo Veillon",
        "an oil painting by Margo Veillon",
        "a watercolor by Margo Veillon",
        "a painting by M. Veillon",
        "a painting by Margo Véillon",
        "a painting in the style of Margo Veillon"
    ]
    art = [
        "a painting",
        "an artwork",
        "an oil painting",
        "a watercolor painting",
        "a fine art painting",
        "a framed painting"
    ]
    other = [
        "a photograph",
        "a logo",
        "a screenshot",
        "an icon",
        "a sketch"
    ]
    negatives = [
        "a painting by Vincent van Gogh",
        "a Van Gogh painting",
        "a post-impressionist painting by Vincent van Gogh",
        "a painting by Claude Monet",
        "a painting by Pablo Picasso"
    ]
    all_prompts = margo + art + other + negatives
    groups = {
        "margo": margo,
        "art": art,
        "other": other,
        "negatives": negatives,
        "all_prompts": all_prompts
    }
    return groups

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
    return {
        "w": int(w),
        "h": int(h),
        "area": int(w*h),
        "std": std_dev,
        "edge_density": edge_density,
        "hue_std": hue_std,
        "avg_color_bgr": avg_color
    }

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

def preprocess_roi_for_clip(roi, trim_border=True, pad=0.02):
    if roi is None or roi.size == 0:
        return None
    img = roi.copy()
    h, w = img.shape[:2]
    if trim_border:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            mask = gray != th
            if np.sum(mask) > 0:
                ys, xs = np.where(mask)
                y0, y1 = max(0, ys.min()-2), min(h, ys.max()+2)
                x0, x1 = max(0, xs.min()-2), min(w, xs.max()+2)
                img = img[y0:y1, x0:x1]
        except Exception:
            pass
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None
    min_side = min(h, w)
    if min_side > 0:
        cy, cx = h//2, w//2
        half = min_side//2
        img = img[cy-half:cy-half+min_side, cx-half:cx-half+min_side]
    if pad > 0 and img.size != 0:
        ph = int(pad * img.shape[0])
        pw = int(pad * img.shape[1])
        img = cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_CONSTANT, value=[255,255,255])
    try:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    except Exception:
        return None
    return pil

def clip_text_image_similarity_preprocessed(roi, prompts):
    """
    Returns (sims, probs) aligned with 'prompts' list.
    sims: cosine similarities (image_feat @ text_feat)
    probs: softmax over logits (sims * logit_scale)
    """
    pil = preprocess_roi_for_clip(roi)
    if pil is None:
        return np.zeros(len(prompts), dtype=float), np.zeros(len(prompts), dtype=float)
    img_t = _preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feat = _model.encode_image(img_t)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        tokens = _tokenizer(prompts)
        tokens = tokens.to(device)
        text_feat = _model.encode_text(tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        sims = (image_feat @ text_feat.T).squeeze(0).cpu().numpy().astype(float)
        try:
            logit_scale = float(_model.logit_scale.exp().cpu().numpy())
        except Exception:
            logit_scale = 100.0
        logits = sims * logit_scale
        # stable softmax
        maxl = np.max(logits)
        exps = np.exp(logits - maxl)
        probs = exps / (np.sum(exps) + 1e-12)
    return sims, probs

# Candidate proposer (contours)
def find_candidate_regions(img, morph_kernel=5, canny1=50, canny2=150, min_area_px=1200, min_area_frac=0.001, max_area_frac=0.95):
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

# -------------------------
# Index functions (unchanged logically)
# -------------------------
def build_index(ref_folder, artist_label="margo_veillon", out_path='veillon_enhanced.pkl'):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required to build index (pip install scikit-learn).")
    files = [f for f in os.listdir(ref_folder) if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))]
    if not files:
        raise ValueError(f"No image files found in {ref_folder}")
    embeddings = []
    fnames = []
    print(f"Building index for {len(files)} images from {ref_folder}")
    for i, f in enumerate(files):
        p = os.path.join(ref_folder, f)
        img = cv2.imread(p)
        if img is None:
            print(f"  Warning: Could not read {f}, skipping")
            continue
        pil = preprocess_roi_for_clip(img)
        if pil is None:
            print(f"  Warning: Could not preprocess {f}, skipping")
            continue
        img_t = _preprocess(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = _model.encode_image(img_t)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.squeeze(0).cpu().numpy().astype(np.float32)
        embeddings.append(emb)
        fnames.append(f)
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            print(f"  Processed {i+1}/{len(files)} images")
    if not embeddings:
        raise RuntimeError("No valid embeddings created; check reference folder and images.")
    emb_mat = np.vstack(embeddings)
    nn = NearestNeighbors(n_neighbors=min(10, len(embeddings)), metric='cosine').fit(emb_mat)
    with open(out_path, 'wb') as fh:
        pickle.dump({
            'filenames': fnames,
            'embeddings': emb_mat,
            'artist_labels': [artist_label] * len(fnames),
            'artist_threshold': 0.895,
            'model_name': 'ViT-B-16',
            'num_items': len(fnames)
        }, fh)
    print(f"\nSaved enhanced index to {out_path}")
    return out_path

def build_multi_artist_index(artist_folders, out_path='multi_artist_index.pkl'):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required to build index.")
    embeddings = []
    fnames = []
    artist_labels = []
    for artist_name, folder_path in artist_folders.items():
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} for artist {artist_name} not found, skipping")
            continue
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))]
        print(f"Processing {len(files)} images for artist: {artist_name}")
        for f in files:
            p = os.path.join(folder_path, f)
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
            artist_labels.append(artist_name)
    if not embeddings:
        raise RuntimeError("No valid embeddings created.")
    emb_mat = np.vstack(embeddings)
    nn = NearestNeighbors(n_neighbors=min(10, len(embeddings)), metric='cosine').fit(emb_mat)
    artist_stats = {}
    unique_artists = set(artist_labels)
    for artist in unique_artists:
        artist_stats[artist] = artist_labels.count(artist)
    with open(out_path, 'wb') as fh:
        pickle.dump({
            'filenames': fnames,
            'embeddings': emb_mat,
            'artist_labels': artist_labels,
            'artist_stats': artist_stats,
            'model_name': 'ViT-B-16',
            'num_items': len(fnames)
        }, fh)
    print(f"\nSaved multi-artist index to {out_path}")
    return out_path

def load_index(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    with open(index_path, 'rb') as f:
        d = pickle.load(f)
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required for NN queries.")
    emb = d['embeddings']
    nn = NearestNeighbors(n_neighbors=min(10, len(emb)), metric='cosine').fit(emb)
    index_obj = {
        'nn': nn,
        'filenames': d['filenames'],
        'embeddings': d['embeddings'],
        'artist_labels': d.get('artist_labels', ['unknown'] * len(d['filenames'])),
        'artist_threshold': d.get('artist_threshold', 0.895),
        'model_name': d.get('model_name', 'unknown'),
        'num_items': len(d['filenames'])
    }
    if 'artist_stats' in d:
        index_obj['artist_stats'] = d['artist_stats']
        index_obj['is_multi_artist'] = True
    else:
        index_obj['is_multi_artist'] = False
    print(f"Loaded index: {index_path} ({index_obj['num_items']} items)")
    return index_obj

def query_index(index_obj, roi, top_k=5, artist_threshold=None):
    if artist_threshold is None:
        artist_threshold = index_obj.get('artist_threshold', 0.895)
    pil = preprocess_roi_for_clip(roi)
    if pil is None:
        return []
    img_t = _preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = _model.encode_image(img_t)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.squeeze(0).cpu().numpy().astype(np.float32)
    actual_top_k = min(top_k, len(index_obj['filenames']))
    dists, idxs = index_obj['nn'].kneighbors(emb.reshape(1, -1), n_neighbors=actual_top_k, return_distance=True)
    results = []
    for dist, idx in zip(dists[0], idxs[0]):
        idx_int = int(idx)
        similarity = 1.0 - float(dist)
        artist_label = index_obj['artist_labels'][idx_int] if idx_int < len(index_obj['artist_labels']) else 'unknown'
        is_margo = False
        if index_obj.get('is_multi_artist', False):
            is_margo = 'margo' in artist_label.lower() and similarity >= artist_threshold
        else:
            is_margo = similarity >= artist_threshold
        results.append({
            'filename': index_obj['filenames'][idx_int],
            'distance': float(dist),
            'similarity': similarity,
            'artist': artist_label,
            'is_margo_veillon': is_margo,
            'above_threshold': similarity >= artist_threshold
        })
    return results

# -------------------------
# Main scanning logic: improved decision rule
# -------------------------
def run_on_scans(scans_folder, out_folder, # thresholds below
                 art_threshold=0.20, margo_threshold=0.22, delta=0.05,
                 negative_gap=0.08, prob_threshold=0.40,
                 artist_threshold=0.895,
                 save_crops=True, visualize=True, logos_folder="logos",
                 debug=False, index_obj=None,
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
    groups = grouped_prompts()
    prompts = groups['all_prompts']
    results = {}

    print(f"Processing {len(scans)} scan(s) from {scans_folder}")
    if index_obj:
        print(f"Using index with {index_obj['num_items']} items (artist_threshold={artist_threshold:.3f})")
    if debug:
        print(f"Thresholds -> art:{art_threshold:.3f} margo:{margo_threshold:.3f} delta:{delta:.3f} neg_gap:{negative_gap:.3f} prob:{prob_threshold:.3f}")

    # indices for groups in flattened prompts
    margo_idxs = list(range(0, len(groups['margo'])))
    art_idxs = list(range(len(groups['margo']), len(groups['margo']) + len(groups['art'])))
    other_idxs = list(range(len(groups['margo']) + len(groups['art']), len(groups['margo']) + len(groups['art']) + len(groups['other'])))
    neg_start = len(groups['margo']) + len(groups['art']) + len(groups['other'])
    neg_idxs = list(range(neg_start, neg_start + len(groups['negatives'])))

    for scan in scans:
        img_path = os.path.join(scans_folder, scan)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read: {img_path}")
            continue

        vis = img.copy()
        candidates = find_candidate_regions(img)
        print(f"[{scan}] Found {len(candidates)} candidate regions")

        image_results = []
        for i, (x, y, rw, rh) in enumerate(candidates):
            roi = img[y:y+rh, x:x+rw]
            met = candidate_metrics(roi)

            # Prefilter by texture + area
            if met.get('area', 0) < prefilter_min_area or met.get('std', 0) < prefilter_std:
                if debug:
                    cv2.imwrite(os.path.join(skipped_folder, f"{Path(scan).stem}_cand{i+1}_prefail.jpg"), roi)
                    print(f"  [{scan}] cand{i+1}: prefilter fail area={met.get('area')} std={met.get('std'):.2f}")
                continue

            if is_known_logo(roi, templates_folder=logos_folder):
                if debug:
                    cv2.imwrite(os.path.join(skipped_folder, f"{Path(scan).stem}_cand{i+1}_logo.jpg"), roi)
                    print(f"  [{scan}] cand{i+1}: logo filtered")
                continue

            if is_text_heavy_by_ocr(roi):
                if debug:
                    cv2.imwrite(os.path.join(skipped_folder, f"{Path(scan).stem}_cand{i+1}_text.jpg"), roi)
                    print(f"  [{scan}] cand{i+1}: text-heavy filtered")
                continue

            # CLIP scoring
            try:
                sims, probs = clip_text_image_similarity_preprocessed(roi, prompts)
            except Exception as e:
                print(f"CLIP error for {scan} cand{i+1}: {e}")
                if debug:
                    cv2.imwrite(os.path.join(skipped_folder, f"{Path(scan).stem}_cand{i+1}_clipfail.jpg"), roi)
                continue

            # group scores
            margo_score = float(np.max(sims[margo_idxs])) if margo_idxs else 0.0
            margo_prob = float(np.max(probs[margo_idxs])) if margo_idxs else 0.0
            art_score = float(np.max(sims[art_idxs])) if art_idxs else 0.0
            other_score = float(np.max(sims[other_idxs])) if other_idxs else 0.0
            neg_score = float(np.max(sims[neg_idxs])) if neg_idxs else 0.0

            # Max of non-margo positive prompts to compute relative gap
            # non_margo_pos_idxs = art + other
            non_margo_pos = float(max(art_score, other_score))

            decision = False
            decision_reasons = []

            # Primary checks:
            # 1) must be artwork-like
            if art_score >= art_threshold:
                decision_reasons.append('artwork_ok')
            else:
                decision_reasons.append('artwork_low')

            # 2) margo relative to other positive prompts
            if (margo_score - non_margo_pos) >= delta and margo_score >= margo_threshold:
                decision = True
                decision_reasons.append('margo_cosine_delta')
            # 3) confident softmax probability for any margo prompt
            if margo_prob >= prob_threshold:
                decision = True
                decision_reasons.append('margo_prob_confident')
            # 4) explicit negative artist gap: margo_score should exceed any negative artist prompt by negative_gap
            if (margo_score - neg_score) < negative_gap:
                # if margo doesn't beat negatives by required gap, penalize
                if debug:
                    decision_reasons.append('neg_artist_close')
                decision = False

            # require artworkness at least (even if margo signal strong)
            if art_score < art_threshold:
                decision = False
                if 'artwork_low' not in decision_reasons:
                    decision_reasons.append('artwork_low')

            entry = {
                "candidate": i+1,
                "coords": [int(x), int(y), int(rw), int(rh)],
                "margo_score": margo_score,
                "margo_prob": margo_prob,
                "art_score": art_score,
                "other_score": other_score,
                "neg_score": neg_score,
                "decision": bool(decision),
                "decision_reasons": decision_reasons
            }

            # Index NN confirmation (strong positive signal)
            if index_obj is not None:
                try:
                    matches = query_index(index_obj, roi, top_k=3, artist_threshold=artist_threshold)
                    if matches:
                        best = matches[0]
                        entry['nn_best_similarity'] = best['similarity']
                        entry['nn_artist'] = best['artist']
                        entry['nn_is_margo'] = best['is_margo_veillon']
                        entry['nn_matches'] = [
                            { 'filename': m['filename'], 'similarity': m['similarity'], 'artist': m['artist'] }
                            for m in matches[:3]
                        ]
                        if best['is_margo_veillon']:
                            # strong override to accept
                            entry['decision'] = True
                            if 'artist_match' not in entry['decision_reasons']:
                                entry['decision_reasons'].append('artist_match')
                        else:
                            # explicit mismatch: weaken decision
                            if 'artist_mismatch' not in entry['decision_reasons']:
                                entry['decision_reasons'].append('artist_mismatch')
                            # keep decision as is (do not force accept)
                    else:
                        entry['decision_reasons'].append('no_index_matches')
                except Exception as e:
                    if debug:
                        print(f"Index query failed for {scan} cand{i+1}: {e}")
                    entry['decision_reasons'].append('index_error')

            # Visualization label and saving
            label = f"M:{entry['margo_score']:.2f}/A:{entry['art_score']:.2f}"
            if 'nn_best_similarity' in entry:
                label += f"/NN:{entry['nn_best_similarity']:.2f}"
            color = (0,255,0) if entry['decision'] else (0,0,255)
            cv2.rectangle(vis, (x,y), (x+rw, y+rh), color, 2)
            cv2.putText(vis, label, (x, max(10, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Save crops or skipped
            if entry['decision'] and save_crops:
                fname = f"{Path(scan).stem}_cand{i+1}_M{entry['margo_score']:.2f}"
                if 'nn_best_similarity' in entry:
                    fname += f"_NN{entry['nn_best_similarity']:.2f}"
                fname += ".jpg"
                cv2.imwrite(os.path.join(crops_folder, fname), roi)
            elif debug:
                fname = f"{Path(scan).stem}_cand{i+1}_rej_M{entry['margo_score']:.2f}.jpg"
                cv2.imwrite(os.path.join(skipped_folder, fname), roi)

            if debug:
                print(f"  [{scan}] cand{i+1}: margo={margo_score:.4f} margo_prob={margo_prob:.4f} art={art_score:.4f} neg={neg_score:.4f} decision={entry['decision']} reasons={entry['decision_reasons']}")

            image_results.append(entry)

        # save visualization (jpg)
        if visualize:
            vis_out = os.path.join(vis_folder, f"vis_{Path(scan).stem}.jpg")
            cv2.imwrite(vis_out, vis)

        accepted = sum(1 for e in image_results if e['decision'])
        print(f"  Accepted: {accepted}/{len(image_results)} candidates")
        results[scan] = image_results

    report = os.path.join(out_folder, "clip_scan_report.json")
    with open(report, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2)

    print("Processing Complete")
    print(f"Report saved to: {report}")
    print(f"Crops folder: {crops_folder}")
    print(f"Visualizations: {vis_folder}")
    print(f"Skipped candidates: {skipped_folder}")

    total_candidates = sum(len(v) for v in results.values())
    total_accepted = sum(sum(1 for e in v if e['decision']) for v in results.values())
    print(f"Total candidates processed: {total_candidates}")
    print(f"Total accepted: {total_accepted}")

    return results

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP artwork scanner with improved prompts/decision")
    parser.add_argument("--scans", default="scans", help="Input folder containing scans")
    parser.add_argument("--out", default="clip_results", help="Output folder for results")
    parser.add_argument("--art-threshold", type=float, default=0.20, help="Artworkness (generic art) threshold")
    parser.add_argument("--margo-threshold", type=float, default=0.22, help="Margo cosine threshold (zero-shot)")
    parser.add_argument("--delta", type=float, default=0.05, help="Margo - other_pos delta")
    parser.add_argument("--negative-gap", type=float, default=0.08, help="Margo - negative-artist gap (must exceed)")
    parser.add_argument("--prob-threshold", type=float, default=0.40, help="Softmax probability threshold for Margo prompts")
    parser.add_argument("--artist-threshold", type=float, default=0.895, help="Index NN artist similarity threshold")
    parser.add_argument("--build-index", default=None, help="Path to reference images folder to build index")
    parser.add_argument("--artist-label", default="margo_veillon", help="Artist label for index")
    parser.add_argument("--index-out", default="veillon_enhanced.pkl", help="Where to save built index")
    parser.add_argument("--build-multi-index", default=None, help="Build multi-artist index as 'name=folder name2=folder2'")
    parser.add_argument("--index", default=None, help="Path to prebuilt index .pkl to use for NN lookup")
    parser.add_argument("--logos-folder", default="logos", help="Folder with small logo templates to ignore")
    parser.add_argument("--prefilter-std", type=float, default=12.0, help="Std-dev prefilter to remove text/flat regions")
    parser.add_argument("--prefilter-min-area", type=int, default=1500, help="Min area prefilter")
    parser.add_argument("--no-visualize", action="store_true", help="Don't write visualizations")
    parser.add_argument("--no-save-crops", action="store_true", help="Don't save accepted crop images")
    parser.add_argument("--debug", action="store_true", help="Save rejected candidates and print debug info")

    args = parser.parse_args()

    index_obj = None

    if args.build_multi_index:
        artist_folders = {}
        for pair in args.build_multi_index.split():
            if '=' in pair:
                artist, folder = pair.split('=', 1)
                artist_folders[artist] = folder
            else:
                print(f"Warning: Invalid format for pair: {pair}")
        if artist_folders:
            build_multi_artist_index(artist_folders, out_path=args.index_out)
        else:
            print("No valid pairs for multi-index; exiting")
            sys.exit(1)

    if args.build_index:
        build_index(args.build_index, artist_label=args.artist_label, out_path=args.index_out)

    if args.index:
        try:
            index_obj = load_index(args.index)
        except Exception as e:
            print(f"Error loading index: {e}")
            sys.exit(1)

    if args.scans and os.path.exists(args.scans):
        run_on_scans(
            scans_folder=args.scans,
            out_folder=args.out,
            art_threshold=args.art_threshold,
            margo_threshold=args.margo_threshold,
            delta=args.delta,
            negative_gap=args.negative_gap,
            prob_threshold=args.prob_threshold,
            artist_threshold=args.artist_threshold,
            save_crops=not args.no_save_crops,
            visualize=not args.no_visualize,
            logos_folder=args.logos_folder,
            debug=args.debug,
            index_obj=index_obj,
            prefilter_std=args.prefilter_std,
            prefilter_min_area=args.prefilter_min_area
        )
    else:
        if args.build_index or args.build_multi_index:
            pass
        else:
            print("Error: No scans folder specified or folder does not exist")
            parser.print_help()
            sys.exit(1)