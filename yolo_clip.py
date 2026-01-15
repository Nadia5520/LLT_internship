#!/usr/bin/env python3
"""
YOLOv8 + CLIP Pipeline - Fixed metadata column issue
"""
import os
import cv2
import torch
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
import json
import argparse
import open_clip
from sklearn.neighbors import NearestNeighbors

class MargoArtDetector:
    def __init__(self, yolo_model="models/yolov8_veillon/weights/best.pt", 
                 clip_index="models/veillon_index.pkl", 
                 metadata_csv="metadata.csv"):
        
        print(f"Loading YOLO model from: {yolo_model}")
        self.yolo = YOLO(yolo_model)
        
        print(f"Loading CLIP index from: {clip_index}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.model.to(self.device).eval()
        
        # Load CLIP index
        with open(clip_index, 'rb') as f:
            index_data = pickle.load(f)
        self.ref_filenames = index_data['filenames']
        self.ref_embeddings = index_data['embeddings']
        
        # Load metadata if exists - with flexible column names
        self.metadata = None
        self.filename_column = None
        
        if os.path.exists(metadata_csv):
            self.metadata = pd.read_csv(metadata_csv)
            print(f"Loaded metadata with columns: {list(self.metadata.columns)}")
            
            # Find the filename column (could be 'filename', 'image_filename', 'file', etc.)
            possible_columns = ['filename', 'image_filename', 'file', 'image', 'name', 'image_name']
            for col in possible_columns:
                if col in self.metadata.columns:
                    self.filename_column = col
                    print(f"Using '{col}' as filename column")
                    break
            
            if self.filename_column is None:
                print(f"WARNING: No filename column found in metadata. Using first column.")
                self.filename_column = self.metadata.columns[0]
        else:
            print(f"WARNING: Metadata file {metadata_csv} not found")
        
        # Build similarity search
        self.nn = NearestNeighbors(n_neighbors=3, metric='cosine').fit(self.ref_embeddings)
    
    def detect_artworks(self, image_path):
        """Detect artwork regions using YOLOv8"""
        results = self.yolo(image_path)
        detections = []
        
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = box.astype(int)
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'crop': None
                    })
        
        return detections
    
    def get_clip_embedding(self, image):
        """Get CLIP embedding for an image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image_t = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image_t)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embedding = embedding.squeeze(0).cpu().numpy()
        
        return embedding
    
    def identify_artwork(self, crop_image, threshold=0.3):
        """Identify if crop matches Margo Veillon artworks"""
        embedding = self.get_clip_embedding(crop_image)
        
        # Find nearest neighbors
        distances, indices = self.nn.kneighbors([embedding])
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            filename = self.ref_filenames[idx]
            similarity = 1.0 - dist  # Convert distance to similarity
            
            # Get metadata
            metadata = {}
            if self.metadata is not None and self.filename_column is not None:
                try:
                    # Find row where filename column matches
                    mask = self.metadata[self.filename_column] == filename
                    if mask.any():
                        metadata = self.metadata[mask].iloc[0].to_dict()
                    else:
                        # Try without extension
                        filename_no_ext = os.path.splitext(filename)[0]
                        mask = self.metadata[self.filename_column].str.contains(filename_no_ext, na=False)
                        if mask.any():
                            metadata = self.metadata[mask].iloc[0].to_dict()
                except Exception as e:
                    print(f"Warning: Could not get metadata for {filename}: {e}")
            
            results.append({
                'filename': filename,
                'similarity': float(similarity),
                'is_margo': similarity >= threshold,
                'metadata': metadata
            })
        
        return results
    
    def process_scan(self, image_path, output_dir="results", threshold=0.3):
        """Process a single scan and identify artworks"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read {image_path}")
            return []
        
        # Detect artworks
        detections = self.detect_artworks(image_path)
        print(f"Found {len(detections)} potential artworks")
        
        results = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Identify artwork
            matches = self.identify_artwork(crop, threshold=threshold)
            best_match = matches[0] if matches else None
            
            # Save crop
            crop_filename = f"{Path(image_path).stem}_art_{i+1}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, crop)
            
            # Prepare result
            result = {
                'scan': Path(image_path).name,
                'crop_id': i+1,
                'crop_file': crop_filename,
                'bbox': [int(x) for x in det['bbox']],
                'confidence': det['confidence'],
                'best_match': best_match,
                'all_matches': matches[:3] if matches else []
            }
            
            results.append(result)
            
            # Print result
            if best_match and best_match['is_margo']:
                print(f"✅ Artwork {i+1}: MATCHES Margo Veillon")
                metadata = best_match['metadata']
                if 'title' in metadata or 'image_filename' in metadata:
                    title = metadata.get('title', metadata.get('image_filename', 'Unknown'))
                    print(f"   Title: {title}")
                if 'artist' in metadata:
                    print(f"   Artist: {metadata['artist']}")
                if 'year' in metadata:
                    print(f"   Year: {metadata.get('year', 'Unknown')}")
                print(f"   Similarity: {best_match['similarity']:.2%}")
                print(f"   Saved as: {crop_filename}")
            elif best_match:
                print(f"❓ Artwork {i+1}: Low similarity ({best_match['similarity']:.2%})")
                print(f"   Best match: {best_match['filename']}")
            else:
                print(f"❌ Artwork {i+1}: No match found")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 + CLIP Artwork Detector")
    parser.add_argument("--scans", default="scans", help="Input scans folder")
    parser.add_argument("--output", default="margo_results", help="Output folder")
    parser.add_argument("--yolo-model", default="models/yolov8_veillon/weights/best.pt", 
                       help="Path to trained YOLO model")
    parser.add_argument("--clip-index", default="models/veillon_index.pkl", 
                       help="Path to CLIP index")
    parser.add_argument("--metadata", default="metadata.csv", 
                       help="Path to metadata CSV")
    parser.add_argument("--threshold", type=float, default=0.3, 
                       help="Similarity threshold for matching (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("MARGO VEILLON ARTWORK DETECTOR")
    print("="*60)
    
    # Initialize detector
    print("\nLoading models...")
    detector = MargoArtDetector(
        yolo_model=args.yolo_model,
        clip_index=args.clip_index,
        metadata_csv=args.metadata
    )
    
    # Process all scans
    scan_files = [f for f in os.listdir(args.scans) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"\nFound {len(scan_files)} scan(s) to process")
    
    all_results = []
    for scan_file in scan_files:
        print(f"\n{'='*60}")
        print(f"Processing: {scan_file}")
        print('='*60)
        
        scan_path = os.path.join(args.scans, scan_file)
        results = detector.process_scan(scan_path, output_dir=args.output, threshold=args.threshold)
        all_results.extend(results)
    
    # Save final report
    report_path = os.path.join(args.output, "detection_report.json")
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("COMPLETE! Results saved to:")
    print(f"  Output folder: {args.output}/")
    print(f"  Report file: {report_path}")
    print(f"  Total artworks detected: {len(all_results)}")
    
    # Summary
    if all_results:
        margo_count = sum(1 for r in all_results 
                          if r['best_match'] and r['best_match']['is_margo'])
        print(f"  Margo Veillon artworks: {margo_count}/{len(all_results)}")
    
    print('='*60)

if __name__ == "__main__":
    main()