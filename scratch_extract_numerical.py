import os
import cv2
import numpy as np

data_dir = 'yolo_dataset/val'
output_file = 'Real_Numerical_Data.md'

if not os.path.exists(data_dir):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('# Error\nDataset validation directory not found.')
    exit(0)

def calculate_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return entropy

classes = ['glaucoma', 'non_glaucoma', 'glaucoma_suspect']

lines = []
lines.append('<div align="center">\n')
lines.append('# Extracted Numerical Tensor Data Matrix (Sample)\n</div>\n')
lines.append('Deep learning networks do not interpret qualitative terms like "Hemorrhage" or "Cupping". Instead, they process massive multi-dimensional float arrays representing light, contrast, and structure anomalies. The table below represents the **real, raw mathematical float data values** extracted directly from our validation fundus image pixels. \n\nThis acts as the fundamental numerical data the models compute prior to finalizing a classification vector.\n')
lines.append('| File Name | Target Label | MegaPixels | Mean Red | Mean Green | Mean Blue | Brightness Variance (Contrast) | Shannon Entropy | Structural Edge Density |')
lines.append('| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |')

count = 0
for cls in classes:
    cls_path = os.path.join(data_dir, cls)
    if not os.path.exists(cls_path): continue
    images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Sample up to 10 images per class
    for img_name in images[:10]:
        img_path = os.path.join(cls_path, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Calculations
        h, w = img.shape[:2]
        mp = (h * w) / 1000000.0
        b, g, r = cv2.split(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)
        variance = np.var(gray)
        entropy = calculate_entropy(gray)
        
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges / 255.0) / (h * w)
        
        filename_short = img_name if len(img_name) < 20 else img_name[:17] + '...'
        
        lines.append(f'| `{filename_short}` | {cls} | {mp:.2f} MP | {mean_r:.2f} | {mean_g:.2f} | {mean_b:.2f} | {variance:.2f} | {entropy:.3f} | {edge_density:.5f} |')
        count += 1

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'Successfully processed {count} images into {output_file}')
