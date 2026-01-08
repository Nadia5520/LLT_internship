#!/usr/bin/env python3
"""
clip_scan_tester.py

Run CLIP-based zero-shot checks over candidate regions found in each image
in a scans folder. Produces a per-image report, visualization images showing
candidate boxes + CLIP scores, and optional saved accepted crops.

Usage:
    python clip_scan_tester.py --scans scans --out clip_results --threshold 0.25 --delta 0.08

Requirements:
    pip install torch torchvision transformers pillow opencv-python numpy

Notes:
 - If you already created clip_check.py (as you showed), this script will import and reuse
   its clip_text_image_similarity function. Otherwise it will load CLIP itself.
 - GPU (CUDA) speeds this up a lot. If you have CUDA, PyTorch should be installed with GPU support.
"""
import os
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from PIL import Image

# Try to import your clip_check helper if present
try:
    from clip_check import clip_text_image_similarity, model as _model, processor as _processor
    HAVE_CLIP_HELPER = True
except Exception:
    HAVE_CLIP_HELPER = False
    clip_text_image_similarity = None

# Lazy import transformers/torch if helper not present
if not HAVE_CLIP_HELPER:
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading CLIP model on", device, " (this may take a moment)...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()

        def clip_text_image_similarity(cv2_bgr_image, text_prompts=["a painting by Margo Veillon"]):
            # convert to PIL RGB
            img_rgb = cv2.cvtColor(cv2_bgr_image, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            inputs = clip_processor(text=text_prompts, images=pil, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = clip_model(**inputs)
                image_emb = outputs.image_embeds   # (1, D)
                text_emb = outputs.text_emb       # (len(prompts), D)
                image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
                text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
                sims = (image_emb @ text_emb.T).squeeze(0).cpu().numpy()
            return sims
    except Exception as e:
        raise RuntimeError("CLIP not available - install transformers & torch or provide clip_check.py") from e

# Candidate region proposer (simple Canny+contours; tuned for artwork crops)
def find_candidate_regions(img, morph_kernel=5, canny1=50, canny2=150, min_area_px=3000, min_area_frac=0.002, max_area_frac=0.9):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, canny1, canny2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    image_area = w * h
    for cnt in contours:
        area = cv2.contourArea(cnt)
        min_area = max(min_area_px, int(min_area_frac * image_area))
        if area < min_area or area > max_area_frac * image_area:
            continue
        x, y, rw, rh = cv2.boundingRect(cnt)
        pad = int(0.01 * max(rw, rh))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(w, x + rw + pad); y1 = min(h, y + rh + pad)
        regions.append((x0, y0, x1 - x0, y1 - y0))
    # merge simple overlaps
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

def pretty_prompt_list():
    # primary prompt is first (Margo); others act as negatives/alternatives
    return [
        "a painting by Margo Veillon",
        "a painting",
        "a photograph",
        "a logo",
        "a screenshot",
        "an icon"
    ]

def run_on_scans(scans_folder, out_folder, threshold=0.25, delta=0.08, save_crops=True, visualize=True):
    ensure = lambda p: os.makedirs(p, exist_ok=True)
    ensure(out_folder)
    crops_folder = os.path.join(out_folder, "crops")
    vis_folder = os.path.join(out_folder, "visualizations")
    ensure(crops_folder); ensure(vis_folder)

    scans = [f for f in os.listdir(scans_folder) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
    results = {}
    prompts = pretty_prompt_list()

    for scan in scans:
        img_path = os.path.join(scans_folder, scan)
        img = cv2.imread(img_path)
        if img is None:
            print("Could not read:", img_path); continue
        h, w = img.shape[:2]
        vis = img.copy()
        candidates = find_candidate_regions(img)
        print(f"[{scan}] {len(candidates)} candidates found")
        image_results = []
        for i, (x, y, rw, rh) in enumerate(candidates):
            roi = img[y:y+rh, x:x+rw]
            try:
                sims = clip_text_image_similarity(roi, prompts)
            except Exception as e:
                print("CLIP error on candidate:", e)
                sims = np.zeros(len(prompts), dtype=float)
            mv_score = float(sims[0])
            other_max = float(max(sims[1:])) if len(sims) > 1 else 0.0
            decision = (mv_score - other_max >= delta) and (mv_score >= threshold)
            entry = {
                "candidate_idx": i+1,
                "coords": [int(x), int(y), int(rw), int(rh)],
                "scores": {p: float(s) for p, s in zip(prompts, sims.tolist())},
                "mv_score": mv_score,
                "other_max": other_max,
                "decision": bool(decision)
            }
            image_results.append(entry)

            label = f"MV:{mv_score:.2f}"
            color = (0,255,255) if decision else (0,0,255)
            cv2.rectangle(vis, (x,y), (x+rw, y+rh), color, 2)
            cv2.putText(vis, label, (x, max(10, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if decision and save_crops:
                outname = f"{Path(scan).stem}_cand{i+1}_mv{mv_score:.2f}.jpg"
                cv2.imwrite(os.path.join(crops_folder, outname), roi)

        # save visualization
        if visualize:
            cv2.imwrite(os.path.join(vis_folder, f"vis_{scan}"), vis)

        # summary print
        accepted = sum(1 for e in image_results if e["decision"])
        print(f"  Accepted: {accepted}/{len(image_results)} (threshold={threshold} delta={delta})")
        results[scan] = image_results

    # write JSON report
    report_path = os.path.join(out_folder, "clip_scan_report.json")
    with open(report_path, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2)
    print("Report saved to", report_path)
    print("Crops saved to", crops_folder)
    print("Visualizations saved to", vis_folder)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scans", default="scans", help="Folder with input images")
    p.add_argument("--out", default="clip_results", help="Output folder")
    p.add_argument("--threshold", type=float, default=0.25, help="Minimum MV similarity to accept")
    p.add_argument("--delta", type=float, default=0.08, help="MV_score - max(other_scores) must be >= delta")
    p.add_argument("--no-save-crops", action="store_true", help="Don't save accepted crop images")
    p.add_argument("--no-visualize", action="store_true", help="Don't save visualization images")
    args = p.parse_args()
    run_on_scans(args.scans, args.out, threshold=args.threshold, delta=args.delta,
                 save_crops=not args.no_save_crops, visualize=not args.no_visualize)