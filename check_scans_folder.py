import os

print("=" * 50)
print("CHECKING SCANS FOLDER")
print("=" * 50)

scans_folder = "scans"

if os.path.exists(scans_folder):
    print("[OK] Found 'scans' folder")
    
    files = os.listdir(scans_folder)
    print(f"Total files: {len(files)}")
    
    # Show all files
    for i, f in enumerate(files, 1):
        file_path = os.path.join(scans_folder, f)
        size_kb = os.path.getsize(file_path) / 1024 if os.path.exists(file_path) else 0
        print(f"{i:2}. {f} ({size_kb:.1f} KB)")
    
    # Count by type
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    pdf_exts = ('.pdf',)
    
    images = [f for f in files if f.lower().endswith(image_exts)]
    pdfs = [f for f in files if f.lower().endswith(pdf_exts)]
    
    print(f"\nSummary:")
    print(f"  Images: {len(images)} files")
    print(f"  PDFs: {len(pdfs)} files")
    print(f"  Other: {len(files) - len(images) - len(pdfs)} files")
    
    if len(images) == 0 and len(pdfs) == 0:
        print("\n[WARNING] No image or PDF files found!")
    
else:
    print("[ERROR] 'scans' folder not found!")
    print("Please create a 'scans' folder and add your scanned documents.")