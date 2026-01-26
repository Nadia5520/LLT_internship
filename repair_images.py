import pandas as pd
from pathlib import Path
import requests

IMAGES_DIR = Path("margo_data/images")
CSV_PATH = Path("margo_data/artworks.csv")

def is_html_file(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            head = f.read(2048).lower()
        return b"<html" in head or b"<!doctype html" in head
    except Exception:
        return True

def download_image(url: str, out_path: Path) -> bool:
    try:
        r = requests.get(url, timeout=30, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        ct = (r.headers.get("Content-Type") or "").lower()
        if r.status_code != 200 or not ct.startswith("image/"):
            print(f"  ❌ Not an image for {out_path.name}: status={r.status_code}, content-type={ct}, url={url}")
            return False

        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        tmp.replace(out_path)
        return True
    except Exception as e:
        print(f"  ❌ Download error for {out_path.name}: {e}")
        return False

def main():
    df = pd.read_csv(CSV_PATH)
    bad = [p for p in IMAGES_DIR.glob("*.jpg") if is_html_file(p)]
    print(f"Found {len(bad)} bad 'jpg' files that are actually HTML.")

    # Map item_id -> image_url
    df["item_id"] = df["item_id"].astype(int)
    url_by_id = dict(zip(df["item_id"], df["image_url"]))

    repaired = 0
    for p in bad:
        item_id = int(p.stem)  # "0273" -> 273
        url = url_by_id.get(item_id)
        if not url or not isinstance(url, str) or not url.strip():
            print(f"  ⚠️ No image_url in CSV for {p.name} (item_id={item_id})")
            continue

        print(f"Re-downloading {p.name} from {url}")
        ok = download_image(url, p)
        if ok:
            repaired += 1
        else:
            # leave the bad file so you can inspect later
            pass

    print(f"Repaired {repaired}/{len(bad)} bad files.")

if __name__ == "__main__":
    main()
