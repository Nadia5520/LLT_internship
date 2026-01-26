import csv
import json
import re
import time
import urllib.parse
import argparse
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE_ITEM_URL = "https://digitalcollections.aucegypt.edu/digital/collection/p15795coll40/id/{}/"

OUTPUT_DIR = Path("margo_data")
IMG_DIR = OUTPUT_DIR / "images"
CSV_PATH = OUTPUT_DIR / "artworks.csv"

IMG_DIR.mkdir(parents=True, exist_ok=True)


def fetch_item(item_id):
    url = BASE_ITEM_URL.format(item_id)
    try:
        resp = requests.get(url, timeout=15)
        print(f"    Status: {resp.status_code} for {url}")
        if resp.status_code != 200:
            return None
        return resp.text
    except Exception as e:
        print(f"    Fetch error: {e}")
        return None


def extract_json_metadata(html):
    """Extract metadata from CONTENTdm's __INITIAL_STATE__ (robust) + HTML fallback."""
    metadata: dict[str, str] = {}

    # The page stores the entire record in a JS string:
    #   window.__INITIAL_STATE__ = JSON.parse("{\\"intl\\":...}");
    # Your previous regex either:
    # - failed because it stopped at the first quote inside the JSON string, or
    # - grabbed multiple <script> blocks (GA script + INITIAL_STATE), causing JSONDecodeError.
    state_pat = r'window\.__INITIAL_STATE__\s*=\s*JSON\.parse\(\s*"((?:\\.|[^"\\])*)"\s*\)'
    m = re.search(state_pat, html, re.DOTALL | re.IGNORECASE)
    if not m:
        print("    ❌ Could not locate window.__INITIAL_STATE__ JSON.parse(\"...\")")
        _html_fallback(html, metadata)
        return metadata

    raw_js_string = m.group(1)
    try:
        # 1) Turn the captured JS string literal into a real Python string (unescape \" \uXXXX etc.)
        json_text = json.loads(f'"{raw_js_string}"')
        # 2) Parse the resulting JSON text into a dict
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"    ⚠️ JSONDecodeError while parsing __INITIAL_STATE__: {e}")
        _html_fallback(html, metadata)
        return metadata

    # CONTENTdm structure for items is very stable:
    # data['item']['item']['fields'] = [{key,label,value,...}, ...]
    fields = None
    try:
        fields = data["item"]["item"]["fields"]
    except Exception:
        # fallback: search for a dict that actually looks like a fields list
        def find_fields(node):
            if isinstance(node, dict):
                if "fields" in node and isinstance(node["fields"], list):
                    sample = node["fields"]
                    if sample and isinstance(sample[0], dict) and {"key", "value"}.issubset(sample[0].keys()):
                        return sample
                for v in node.values():
                    out = find_fields(v)
                    if out is not None:
                        return out
            elif isinstance(node, list):
                for it in node:
                    out = find_fields(it)
                    if out is not None:
                        return out
            return None

        fields = find_fields(data)

    if isinstance(fields, list):
        for f in fields:
            key = (f.get("key") or "").strip().lower()
            val = (f.get("value") or "").strip()
            if key and val:
                metadata[key] = val
        print(f"    ✅ JSON fields extracted: {len(metadata)}")
    else:
        print("    ⚠️ No fields list found in __INITIAL_STATE__")

    _html_fallback(html, metadata)
    return metadata

def _html_fallback(html, metadata):
    """Robust HTML extraction"""
    print("    🔍 HTML fallback...")
    soup = BeautifulSoup(html, 'html.parser')
    
    # Meta tags
    for meta in soup.find_all('meta', attrs={'name': True, 'content': True}):
        name = meta['name'].lower()
        content = meta['content'].strip()
        if content:
            metadata.setdefault(name, content)

    # OpenGraph meta tags (og:title, og:description, og:image, ...)
    for meta in soup.find_all('meta', attrs={'property': True, 'content': True}):
        prop = meta['property'].lower()
        content = meta['content'].strip()
        if content:
            metadata.setdefault(prop, content)
    
    # Common CONTENTdm patterns
    patterns = {
        r'dc:title': 'title',
        r'dc:creator': 'creator', 
        r'dc:description': 'description'
    }
    
    for pat, key in patterns.items():
        matches = soup.find_all(attrs={'class': re.compile(pat, re.I)})
        for el in matches:
            text = el.get_text(strip=True)
            if text:
                metadata[key] = text
                print(f"      HTML {key}: {text[:50]}...")
                break
    
    print(f"    ✅ HTML metadata keys total now: {len(metadata)}")




def get_image_url(html):
    # og:image first
    match = re.search(r'<meta property="og:image" content="([^"]+)"', html)
    if match:
        return match.group(1)
    
    # JSON imageUri
    match = re.search(r'"imageUri":"([^"]+)"', html)
    if match:
        return match.group(1)
    
    print("    ⚠️ No image URL found")
    return ""


def download_image(image_url, item_id):
    if not image_url:
        return ""
    
    fname = f"{item_id:04d}.jpg"
    path = IMG_DIR / fname
    
    if path.exists():
        print(f"    📁 Image exists")
        return str(path)
    
    try:
        print(f"    ⬇️  Downloading {image_url}")
        resp = requests.get(image_url, timeout=15, stream=True)
        if resp.status_code == 200:
            with open(path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            print(f"    ✅ Saved {fname}")
            return str(path)
        else:
            print(f"    ❌ Image HTTP {resp.status_code}")
    except Exception as e:
        print(f"    ❌ Download error: {e}")
    return ""


def test_single_item(item_id=121):
    print(f"\n{'='*60}")
    print(f"🧪 TESTING ITEM ID {item_id}")
    print(f"{'='*60}")
    
    html = fetch_item(item_id)
    if not html:
        print("❌ PAGE NOT FOUND")
        return False
    
    print(f"📄 HTML: {len(html)} chars | First 200: {html[:200]}")
    
    meta = extract_json_metadata(html)
    img_url = get_image_url(html)
    img_path = download_image(img_url, item_id)
    print(f"\n  🐛 DEBUG ID {item_id}:")
    print(f"    All keys ({len(meta)}): {list(meta.keys())}")

    # Raw fields (CONTENTdm keys)
    print(f"    covera (raw): '{meta.get('covera', 'MISSING')}'")
    print(f"    type   (raw): '{meta.get('type', 'MISSING')}'")

    # Build the same record mapping used for CSV
    record = {
        "item_id": item_id,
        "title": meta.get('title', f'Art {item_id}'),
        "creator": meta.get('creata', 'Margo Veillon'),
        "date": meta.get('datea', meta.get('date', '')),
        "collection": meta.get('collec', 'Margo Veillon Art Collection'),
        "material": meta.get('covera', meta.get('material', '')),
        "technique": meta.get('type', meta.get('technique', '')),
        "worktype": meta.get('medium', meta.get('worktype', '')),
        "description": meta.get('descri', meta.get('description', ''))[:300],
        "location": meta.get('locati', meta.get('location', '')),
        "detail_url": BASE_ITEM_URL.format(item_id),
        "image_url": img_url,
        "local_image_path": img_path,
    }

    # Final values (what goes to CSV)
    print(f"    material (final): '{record.get('material', 'MISSING')}'")
    print(f"    technique(final): '{record.get('technique', 'MISSING')}'")

    print(f"    collec (raw): '{meta.get('collec', 'MISSING')}'")
    print(f"    Raw meta sample: {dict(list(meta.items())[:3])}")

    for k, v in sorted(meta.items()):
        print(f"  {k:12}: {v[:60]}...")
    
    print(f"\n🖼️ Image: {img_path or 'FAILED'}")
    return bool(meta and img_path)



def main(start_id: int = 1, end_id: int = 400):
    print("\n🧪 Auto-testing ID 121 first...")
    if not test_single_item(121):
        print("\n❌ Test failed - check debug output above!")
        return
    
    print(f"\n🔥 Starting FULL scraper ({start_id}-{end_id})...")
    
    fieldnames = [
        "item_id",
        "title",
        "creator",
        "date",
        "collection",
        "material",
        "technique",
        "worktype",
        "description",
        "location",
        "detail_url",
        "image_url",
        "local_image_path",
        "raw_metadata",
    ]
    
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        success = 0
        skipped = 0
        
        for item_id in range(start_id, end_id + 1):
            print(f"\n[{success}/{item_id-1}] ID {item_id:3d}", end=" ")
            
            html = fetch_item(item_id)
            if not html:
                print("→ no page")
                skipped += 1
                continue
            
            meta = extract_json_metadata(html)
            img_url = get_image_url(html)
            img_path = download_image(img_url, item_id)
            
            if img_path and meta.get('title', '').strip():  # Require title + image
                record = {
                    "item_id": item_id,
                    "title": meta.get('title', f'Art {item_id}'),
                    "creator": meta.get('creata', 'Margo Veillon'),
                    "date": meta.get('datea', meta.get('date', '')),
                    "collection": meta.get('collec', 'Margo Veillon Art Collection'),
                    "material": meta.get('covera', meta.get('material', '')),
                    "technique": meta.get('type', meta.get('technique', '')),
                    "worktype": meta.get('medium', meta.get('worktype', '')),
                    "description": meta.get('descri', meta.get('description', ''))[:300],
                    "location": meta.get('locati', meta.get('location', '')),
                    "detail_url": BASE_ITEM_URL.format(item_id),
                    "image_url": img_url,
                    "local_image_path": img_path,
                    "raw_metadata": json.dumps(meta, ensure_ascii=False),
                }
                print(f"\n  🐛 DEBUG ID {item_id}:")
                print(f"    covera (raw): '{meta.get('covera', 'MISSING')}'")
                print(f"    type   (raw): '{meta.get('type', 'MISSING')}'")
                print(f"    material (final): '{record.get('material', 'MISSING')}'")
                print(f"    technique(final): '{record.get('technique', 'MISSING')}'")
                print(f"    collec: '{meta.get('collec', 'MISSING')}'")
                
                writer.writerow(record)
                success += 1
                print(f"✅ {record['title'][:35]}...")
            else:
                print("→ no title/image")
                skipped += 1
            
            time.sleep(0.4)  # Polite rate limiting
        
        print(f"\n🎉 FINISHED!")
        print(f"✅ Success: {success} artworks")
        print(f"⏭️  Skipped: {skipped} (no page/title/image)")
        print(f"📁 CSV: {CSV_PATH}")
        print(f"🖼️  Images: {len(list(IMG_DIR.glob('*.jpg')))}")
        print(f"\n🚀 Run: streamlit run app_streamlit_similarity.py")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape AUC Margo Veillon items to CSV + images")
    parser.add_argument("--test", type=int, default=None, help="Test a single item_id (no full scrape)")
    parser.add_argument("--start", type=int, default=1, help="Start item_id (inclusive) for full scrape")
    parser.add_argument("--end", type=int, default=400, help="End item_id (inclusive) for full scrape")
    args = parser.parse_args()

    if args.test is not None:
        test_single_item(args.test)
    else:
        main(start_id=args.start, end_id=args.end)

