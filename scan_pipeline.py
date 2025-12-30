import os
import cv2
import numpy as np

print("=" * 50)
print("IMPROVED ARTWORK DETECTION PIPELINE")
print("=" * 50)

# Check scans folder
if not os.path.exists("scans"):
    print("ERROR: No 'scans' folder")
    exit()

# List files
files = [f for f in os.listdir("scans") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(files)} scan files")

# Clean output folders
for folder in ["detected_pages", "cropped_artworks"]:
    if os.path.exists(folder):
        # Remove old files
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    else:
        os.makedirs(folder)

# Process each file
total_artworks = 0

for file_name in files:
    print(f"\nProcessing: {file_name}")
    
    # Load image
    img = cv2.imread(os.path.join("scans", file_name))
    if img is None:
        print("  ERROR: Could not read file")
        continue
    
    height, width = img.shape[:2]
    print(f"  Size: {width}x{height}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # METHOD 1: Look for areas with paintings (usually have frames/borders)
    # Paintings typically have: dark borders, good contrast, rectangular shape
    
    # Step 1: Detect edges
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    
    # Step 2: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 3: Filter contours
    artwork_boxes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter by size (artworks aren't tiny or huge)
        if area < 8000 or area > 200000:
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter by aspect ratio (artworks are usually reasonable proportions)
        aspect = w / h
        if aspect < 0.4 or aspect > 2.5:
            continue
        
        # Filter by position (artworks usually not at very edges)
        margin = 20
        if x < margin or y < margin or (x + w) > (width - margin) or (y + h) > (height - margin):
            continue
        
        # Check the region content
        roi = img[y:y+h, x:x+w]
        
        # 1. Check color variance (artworks have more colors than UI buttons)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_variance = np.std(roi_hsv[:,:,0])  # Hue variance
        
        # 2. Check if it looks like UI element (Google Translate buttons are usually solid colors)
        if color_variance < 15:  # Low color variance = likely UI button
            continue
        
        # 3. Check if region has frame-like edges (artworks often have borders)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_edges = cv2.Canny(roi_gray, 50, 150)
        edge_density = np.sum(roi_edges > 0) / (w * h)
        
        # Artworks have moderate edge density, UI buttons have very high or very low
        if edge_density < 0.01 or edge_density > 0.3:
            continue
        
        # Passed all filters - likely artwork
        artwork_boxes.append((x, y, w, h))
    
    print(f"  Found {len(artwork_boxes)} potential artwork regions")
    
    # Draw and save results
    if artwork_boxes:
        img_with_boxes = img.copy()
        for (x, y, w, h) in artwork_boxes:
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 3)
            # Label
            cv2.putText(img_with_boxes, f"Artwork", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save detection result
        det_path = os.path.join("detected_pages", f"detected_{file_name}")
        cv2.imwrite(det_path, img_with_boxes)
        print(f"  Saved detection: {det_path}")
        
        # Crop and save artworks
        for i, (x, y, w, h) in enumerate(artwork_boxes):
            artwork = img[y:y+h, x:x+w]
            crop_path = os.path.join("cropped_artworks", f"{file_name[:-4]}_artwork_{i+1}.jpg")
            cv2.imwrite(crop_path, artwork)
            print(f"  Cropped artwork: {crop_path}")
            total_artworks += 1
    else:
        print(f"  No artworks detected")

print("\n" + "=" * 50)
print("SUMMARY:")
print(f"Total artworks extracted: {total_artworks}")
print(f"Detection results: 'detected_pages/' folder")
print(f"Cropped artworks: 'cropped_artworks/' folder")
print("=" * 50)

# Show what we extracted
if total_artworks > 0:
    print("\nExtracted artwork files:")
    for f in os.listdir("cropped_artworks"):
        print(f"  - {f}")