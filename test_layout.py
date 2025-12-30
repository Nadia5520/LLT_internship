import cv2
import layoutparser as lp
import matplotlib.pyplot as plt

# Load a sample page image (from your scans)
image_path = "test_page.jpg"  # Change this to your actual file

try:
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("Testing LayoutParser...")
    model = lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )
    
    layout = model.detect(img_rgb)
    print(f"Found {len(layout)} layout blocks")
    
    # Show what types we found
    for block in layout[:5]:  # First 5 blocks
        print(f"  Type: {block.type}, Coordinates: {block.coordinates}")
    
    print("✅ LayoutParser is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nLet's try a simpler approach first...")