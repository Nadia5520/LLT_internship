import os
import cv2
import numpy as np

print("=" * 60)
print("ARTWORK DETECTION - SIMPLE VERSION")
print("=" * 60)

# Check scans folder
if not os.path.exists("scans"):
    print("ERROR: No 'scans' folder")
    exit()

files = [f for f in os.listdir("scans") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(files)} scan files")

# Create output folders
for folder in ["simple_detections", "simple_artworks"]:
    if os.path.exists(folder):
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    else:
        os.makedirs(folder)

# ARTWORK DETECTION HEURISTICS
def is_likely_artwork(roi):
    """Check if a region looks like an artwork (not UI/text)."""
    if roi.size == 0:
        return False
    
    h, w = roi.shape[:2]
    
    # 1. Check aspect ratio (artworks are usually reasonable)
    aspect = w / h
    if aspect < 0.3 or aspect > 3:
        return False
    
    # 2. Check color variety (artworks have colors, UI/text is limited)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue_std = np.std(hsv[:,:,0])
    sat_std = np.std(hsv[:,:,1])
    
    # UI elements have low color variety
    if hue_std < 10 and sat_std < 20:
        return False
    
    # 3. Check edge pattern (artworks have organic edges, UI has sharp edges)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (w * h)
    
    # Too few edges (solid color) or too many (text/UI)
    if edge_density < 0.005 or edge_density > 0.3:
        return False
    
    # 4. Check for frame/border (artworks often have them)
    # Look for dark border around lighter center
    border_size = min(w, h) // 10
    if border_size > 5:
        # Check top border
        top_region = roi[:border_size, :]
        bottom_region = roi[-border_size:, :]
        left_region = roi[:, :border_size]
        right_region = roi[:, -border_size:]
        
        border_regions = [top_region, bottom_region, left_region, right_region]
        center_region = roi[border_size:-border_size, border_size:-border_size]
        
        if center_region.size > 0:
            border_brightness = np.mean([np.mean(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)) 
                                         for r in border_regions])
            center_brightness = np.mean(cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY))
            
            # If border is darker than center (like a frame)
            if border_brightness < center_brightness * 0.8:
                return True
    
    return True

# Process each file
for file_name in files:
    print(f"\nProcessing: {file_name}")
    
    img = cv2.imread(os.path.join("scans", file_name))
    if img is None:
        continue
    
    height, width = img.shape[:2]
    img_detection = img.copy()
    
    # METHOD: Grid search for artwork-like regions
    cell_size = 200
    stride = 150
    
    detections = []
    
    for y in range(0, height - cell_size, stride):
        for x in range(0, width - cell_size, stride):
            roi = img[y:y+cell_size, x:x+cell_size]
            
            if is_likely_artwork(roi):
                detections.append((x, y, cell_size, cell_size))
    
    print(f"  Initial detections: {len(detections)}")
    
    # Merge nearby detections
    merged = []
    for (x, y, w, h) in detections:
        found = False
        for i, (mx, my, mw, mh) in enumerate(merged):
            # If close to existing detection, merge
            if (abs(x - mx) < 50 and abs(y - my) < 50):
                new_x = min(x, mx)
                new_y = min(y, my)
                new_w = max(x + w, mx + mw) - new_x
                new_h = max(y + h, my + mh) - new_y
                merged[i] = (new_x, new_y, new_w, new_h)
                found = True
                break
        
        if not found:
            merged.append((x, y, w, h))
    
    print(f"  After merging: {len(merged)} regions")
    
    # Draw and save
    for i, (x, y, w, h) in enumerate(merged):
        cv2.rectangle(img_detection, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Crop and save
        artwork = img[y:y+h, x:x+w]
        crop_path = os.path.join("simple_artworks", f"{file_name[:-4]}_art_{i+1}.jpg")
        cv2.imwrite(crop_path, artwork)
    
    if merged:
        det_path = os.path.join("simple_detections", f"det_{file_name}")
        cv2.imwrite(det_path, img_detection)
        print(f"  Saved: {det_path}")

print("\n" + "=" * 60)
print("DONE! Check 'simple_artworks/' folder")
print("=" * 60)