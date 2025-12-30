import os
import cv2
import numpy as np

print("=" * 50)
print("ART COLLECTION - SIMPLE PIPELINE")
print("=" * 50)

# Check scans folder
scans_folder = "scans"
if not os.path.exists(scans_folder):
    print("[ERROR] 'scans' folder not found!")
    exit()

files = os.listdir(scans_folder)
if not files:
    print("[ERROR] No files in 'scans' folder!")
    exit()

print(f"Found {len(files)} file(s) in 'scans' folder:")

# Create output folders
for folder in ["pages", "extracted_artworks", "detection_results"]:
    os.makedirs(folder, exist_ok=True)

# Process first file only (for testing)
test_file = files[0]
print(f"\nTesting with: {test_file}")

file_path = os.path.join(scans_folder, test_file)

# Check if it's an image
if test_file.lower().endswith(('.jpg', '.jpeg', '.png')):
    img = cv2.imread(file_path)
    if img is not None:
        print(f"Image loaded: {img.shape[1]}x{img.shape[0]} pixels")
        
        # Simple detection: find dark/contrasty regions
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Method: find areas with high contrast
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 3000 < area < 100000:  # Reasonable artwork size
                x, y, w, h = cv2.boundingRect(cnt)
                regions.append((x, y, w, h))
        
        print(f"Found {len(regions)} potential artwork regions")
        
        # Draw and save
        img_detected = img.copy()
        for (x, y, w, h) in regions:
            cv2.rectangle(img_detected, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Save results
        det_path = f"detection_results/detection_{test_file}"
        cv2.imwrite(det_path, img_detected)
        print(f"Detection saved: {det_path}")
        
        # Extract artworks
        for i, (x, y, w, h) in enumerate(regions):
            artwork = img[y:y+h, x:x+w]
            art_path = f"extracted_artworks/artwork_{i+1}.jpg"
            cv2.imwrite(art_path, artwork)
        
        print(f"Extracted {len(regions)} artworks to 'extracted_artworks/'")
        
    else:
        print("[ERROR] Could not read image file!")
else:
    print(f"[SKIP] {test_file} is not a supported image format")
    print("Supported: .jpg, .jpeg, .png")

print("\n" + "=" * 50)
print("DONE! Check the output folders.")