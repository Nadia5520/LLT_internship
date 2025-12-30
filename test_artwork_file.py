import cv2
import numpy as np
import os

print("Testing file that should contain artwork...")
file_path = "scans/Margo_Veillon_Painting_Tenderness_1973.PNG"

if os.path.exists(file_path):
    img = cv2.imread(file_path)
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Show a small preview (first 500x500 pixels)
    preview = img[:500, :500]
    cv2.imwrite("preview_top_left.jpg", preview)
    print("Preview saved: preview_top_left.jpg")
    print("Check this file - what do you see? Artwork or text?")
    
    # Check image composition
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Look for framed artwork (dark border around lighter center)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use multiple methods
    print("\nTrying different detection methods:")
    
    # Method 1: Look for rectangular regions
    edges = cv2.Canny(blur, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Method 1 (edges): Found {len(contours)} contours")
    
    # Method 2: Look for color regions
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, sat_thresh = cv2.threshold(saturation, 40, 255, cv2.THRESH_BINARY)
    sat_contours, _ = cv2.findContours(sat_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Method 2 (color): Found {len(sat_contours)} saturated regions")
    
    # Method 3: Simple - look for non-white regions
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    binary_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Method 3 (non-white): Found {len(binary_contours)} dark regions")
    
    # Filter and show largest region
    if binary_contours:
        largest = max(binary_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > 10000:  # Significant size
            x, y, w, h = cv2.boundingRect(largest)
            print(f"\nLargest non-white region:")
            print(f"  Position: ({x}, {y})")
            print(f"  Size: {w}x{h} pixels")
            print(f"  Area: {area:.0f} px²")
            
            # Crop it
            artwork = img[y:y+h, x:x+w]
            cv2.imwrite("potential_artwork.jpg", artwork)
            print(f"\nExtracted: potential_artwork.jpg")
        else:
            print("\nNo large artwork regions found - may be text-only")
    
else:
    print(f"File not found: {file_path}")