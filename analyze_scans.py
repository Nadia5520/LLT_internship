import cv2
import os

print("=" * 60)
print("ANALYZING SCAN CONTENT - FIXED")
print("=" * 60)

scans_folder = "scans"
files = os.listdir(scans_folder)

print("Analysis of first 4 files:\n")

for i, file_name in enumerate(files[:4], 1):
    file_path = os.path.join(scans_folder, file_name)
    img = cv2.imread(file_path)
    
    if img is not None:
        print(f"{i}. {file_name}")
        print(f"   Size: {img.shape[1]}x{img.shape[0]} pixels")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Check edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.countNonZero(edges) / (img.shape[0] * img.shape[1])
        
        if edge_density > 0.1:
            print(f"   Likely: TEXT DOCUMENT (edge density: {edge_density:.3f})")
        else:
            print(f"   Likely: IMAGE/ARTWORK PAGE (edge density: {edge_density:.3f})")
        
        # Check for large regions
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        large_regions = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10000:
                large_regions += 1
        
        print(f"   Large distinct regions: {large_regions}")
        
        # Quick visual check
        if "painting" in file_name.lower() or "studies" in file_name.lower():
            print("   NOTE: Name suggests artwork content")
        
        print()