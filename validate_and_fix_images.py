#!/usr/bin/env python3
"""
validate_and_fix_images.py

Usage:
  python validate_and_fix_images.py veillon_images fixed_veillon_images

- Scans input folder for image files.
- Reports files that cannot be opened/verified.
- Re-saves readable images as JPEG into output folder (normalizes PNG issues).
"""
import os
import sys
from PIL import Image

def main(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    files = [f for f in os.listdir(src_folder) if f.lower().endswith(exts)]
    bad = []
    fixed = 0
    for fn in files:
        src = os.path.join(src_folder, fn)
        try:
            # verify file integrity
            with Image.open(src) as im:
                im.verify()
            # reopen (verify() closes file) and convert+save as JPEG
            with Image.open(src) as im:
                im = im.convert('RGB')
                base = os.path.splitext(fn)[0]
                dst = os.path.join(dst_folder, base + '.jpg')
                im.save(dst, format='JPEG', quality=95)
                fixed += 1
        except Exception as e:
            bad.append((fn, str(e)))
    print(f"Processed {len(files)} files. Re-saved {fixed} files to '{dst_folder}'.")
    if bad:
        print("Files that failed to open/verify:")
        for fn, err in bad:
            print("  ", fn, "-", err)
    else:
        print("No bad files found.")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python validate_and_fix_images.py <input_folder> <output_folder>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])