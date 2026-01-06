import os
import cv2
import numpy as np

print("=" * 60)
print("FINAL ARTWORK EXTRACTOR - WEEK 1 COMPLETE")
print("=" * 60)

# Configuration
SCANS_FOLDER = "scans"
OUTPUT_FOLDER = "extracted_artworks_clean"
VISUALIZATION_FOLDER = "detection_visualizations"

# Create folders
for folder in [OUTPUT_FOLDER, VISUALIZATION_FOLDER]:
    if os.path.exists(folder):
        # Clear old files
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    else:
        os.makedirs(folder)

# Get all scan files
scan_files = [f for f in os.listdir(SCANS_FOLDER) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if not scan_files:
    print(f"❌ No files found in '{SCANS_FOLDER}' folder")
    exit()

print(f"📁 Found {len(scan_files)} scan files")

def is_ui_element(roi):
    """Check if region is a UI element (not artwork)."""
    if roi.size == 0:
        return True
    
    h, w = roi.shape[:2]
    
    # UI elements are usually:
    # 1. Small icons (under 100px)
    if w < 80 or h < 80:
        return True
    
    # 2. Solid colors (low color variance)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue_std = np.std(hsv[:,:,0])
    if hue_std < 5:  # Very uniform color
        return True
    
    # 3. High contrast edges (like text/buttons)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (w * h)
    
    if edge_density > 0.3:  # Too many sharp edges
        return True
    
    # 4. Check for common UI colors (blues, grays)
    avg_color = np.mean(roi, axis=(0, 1))
    # UI buttons often have specific blue shades
    if (avg_color[0] < 100 and avg_color[1] < 100 and avg_color[2] > 150):  # Blue-ish
        return True
    
    return False

def is_likely_artwork(roi):
    """Check if region looks like an artwork."""
    if roi.size == 0:
        return False
    
    h, w = roi.shape[:2]
    
    # Artworks are usually:
    # 1. Reasonable size (not too small, not huge)
    if w < 150 or h < 150 or w > 1000 or h > 1000:
        return False
    
    # 2. Good aspect ratio (not extreme)
    aspect = w / h
    if aspect < 0.3 or aspect > 3:
        return False
    
    # 3. Has visual complexity (not solid color)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    if std_dev < 15:  # Too uniform
        return False
    
    # 4. Not a UI element
    if is_ui_element(roi):
        return False
    
    return True

# Process each scan
total_artworks = 0

for scan_file in scan_files:
    print(f"\n🔍 Processing: {scan_file}")
    
    # Load image
    img_path = os.path.join(SCANS_FOLDER, scan_file)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"  ❌ Could not read file")
        continue
    
    height, width = img.shape[:2]
    print(f"  📐 Size: {width}x{height}")
    
    # Create visualization
    vis_img = img.copy()
    
    # Find candidate regions using multiple methods
    
    # METHOD 1: Edge-based detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    artwork_regions = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter by size
        if area < 10000 or area > 300000:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Check if region looks like artwork
        roi = img[y:y+h, x:x+w]
        if is_likely_artwork(roi):
            artwork_regions.append((x, y, w, h))
    
    print(f"  🎨 Found {len(artwork_regions)} potential artworks")
    
    # Draw and save
    for i, (x, y, w, h) in enumerate(artwork_regions):
        # Draw rectangle
        color = (0, 255, 0)  # Green for artworks
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 3)
        cv2.putText(vis_img, f"Art {i+1}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Extract and save
        artwork = img[y:y+h, x:x+w]
        art_filename = f"{scan_file[:-4]}_artwork_{i+1}.jpg"
        art_path = os.path.join(OUTPUT_FOLDER, art_filename)
        cv2.imwrite(art_path, artwork)
        
        total_artworks += 1
    
    # Save visualization
    vis_path = os.path.join(VISUALIZATION_FOLDER, f"vis_{scan_file}")
    cv2.imwrite(vis_path, vis_img)
    
    if artwork_regions:
        print(f"  💾 Saved {len(artwork_regions)} artworks to '{OUTPUT_FOLDER}'")

print(f"\n{'='*60}")
print(f"✅ WEEK 1 COMPLETE!")
print(f"📊 Results:")
print(f"   Processed: {len(scan_files)} scan files")
print(f"   Extracted: {total_artworks} artwork images")
print(f"   Output folder: '{OUTPUT_FOLDER}'")
print(f"   Visualizations: '{VISUALIZATION_FOLDER}'")
print(f"{'='*60}")

# Show what we got
if total_artworks > 0:
    print("\n📁 Extracted artworks:")
    artworks = os.listdir(OUTPUT_FOLDER)
    for i, art in enumerate(artworks[:5]):  # Show first 5
        print(f"   {i+1}. {art}")
    if len(artworks) > 5:
        print(f"   ... and {len(artworks)-5} more")
else:
    print("\n⚠️  No artworks extracted. Possible issues:")
    print("   1. Scans might be text-only (no images)")
    print("   2. Detection thresholds might be too strict")
    print("   3. Artworks might be too small/large")