#!/usr/bin/env python3
"""
Create a YOLOv8 dataset from a folder of full-artwork images.

Usage:
  python create_yolo_dataset_from_artworks.py --artworks veillon_images --out dataset --val-split 0.2

Behavior:
 - Copies artwork images into dataset/images/train and dataset/images/val (80/20 by default).
 - Writes YOLO label files where each image has a single box: class 0, center=(0.5,0.5), w=0.98, h=0.98.
 - Writes dataset/data.yaml ready for YOLOv8 training.
"""
import os, shutil, random, argparse
from pathlib import Path

def main(args):
    art_folder = args.artworks
    out = args.out
    val_split = args.val_split
    seed = args.seed

    exts = ('.jpg','.jpeg','.png','.bmp','.webp','tif','tiff')
    files = [f for f in os.listdir(art_folder) if f.lower().endswith(exts)]
    if not files:
        raise SystemExit("No artwork files found in " + art_folder)
    random.seed(seed)
    random.shuffle(files)
    n_val = int(len(files) * val_split)
    val_files = files[:n_val]
    train_files = files[n_val:]

    # create dirs
    for p in ["images/train","images/val","labels/train","labels/val"]:
        os.makedirs(os.path.join(out,p), exist_ok=True)

    def copy_and_label(fname, subset):
        src = os.path.join(art_folder, fname)
        dst_img = os.path.join(out, "images", subset, Path(fname).stem + ".jpg")
        # copy and convert to jpg by reading+writing if necessary
        try:
            from PIL import Image
            im = Image.open(src).convert("RGB")
            im.save(dst_img, format="JPEG", quality=95)
        except Exception:
            shutil.copy2(src, dst_img)
        # write label: class 0, center 0.5 0.5 width 0.98 height 0.98
        label_path = os.path.join(out, "labels", subset, Path(dst_img).stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(f"0 0.5 0.5 0.98 0.98\n")

    for f in train_files:
        copy_and_label(f, "train")
    for f in val_files:
        copy_and_label(f, "val")

    # write data.yaml
    import yaml
    data = {
        "train": str(Path(out) / "images" / "train"),
        "val": str(Path(out) / "images" / "val"),
        "nc": 1,
        "names": ["artwork"]
    }
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as yf:
        yaml.dump(data, yf)
    print(f"Dataset created at {out}. Train:{len(train_files)} Val:{len(val_files)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artworks", required=True, help="Folder with full artwork images")
    p.add_argument("--out", default="dataset", help="Output dataset folder")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)