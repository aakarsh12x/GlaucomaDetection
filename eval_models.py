import os
import sys
import torch
import time
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# Add models to sys.path
sys.path.append(os.path.join(os.getcwd(), "models"))
from mamba_out.train import GlaucomaMambaOut
from vision_mamba.train import GlaucomaVim

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "yolo_dataset/val"
MODELS_TO_EVAL = {
    "YOLOv11": {
        "type": "ultralytics",
        "path": "runs/classify/glaucoma_runs/yolo11_glaucoma_classification/weights/best.pt"
    },
    "MambaOut": {
        "type": "pytorch",
        "class": GlaucomaMambaOut,
        "path": "runs/mamba_out/best.pt"
    },
    "Vision Mamba": {
        "type": "pytorch",
        "class": GlaucomaVim,
        "path": "runs/vision_mamba/best.pt"
    }
}

CLASS_MAP = ["glaucoma", "glaucoma_suspect", "non_glaucoma"]

# Transforms for PyTorch
pytorch_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def evaluate_all():
    global_results = {}

    for name, config in MODELS_TO_EVAL.items():
        print(f"\nEvaluating {name}...")
        
        if not os.path.exists(config["path"]):
            print(f"Skipping {name}: Weights not found at {config['path']}")
            continue

        # Load Model
        if config["type"] == "ultralytics":
            model = YOLO(config["path"])
        else:
            model = config["class"](num_classes=len(CLASS_MAP))
            model.load_state_dict(torch.load(config["path"], map_location=DEVICE))
            model.to(DEVICE)
            model.eval()

        # Run Eval
        correct = 0
        total = 0
        per_class = {cls: {"correct": 0, "total": 0} for cls in CLASS_MAP}
        
        start_time = time.time()
        
        for cls in CLASS_MAP:
            cls_dir = os.path.join(DATA_DIR, cls)
            if not os.path.isdir(cls_dir):
                continue
            
            images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in images:
                img_path = os.path.join(cls_dir, img_name)
                img = Image.open(img_path).convert('RGB')
                
                # Inference
                if config["type"] == "ultralytics":
                    results = model.predict(img, imgsz=224, verbose=False)
                    top_idx = results[0].probs.top1
                    # YOLO might have different internal class mapping, verify
                    pred_name = results[0].names[top_idx].lower().replace(" ", "_")
                else:
                    img_tensor = pytorch_transforms(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        top_idx = torch.argmax(outputs[0]).item()
                        pred_name = CLASS_MAP[top_idx]
                
                # Update metrics
                total += 1
                per_class[cls]["total"] += 1
                
                # Normalize names for comparison
                if pred_name == cls:
                    correct += 1
                    per_class[cls]["correct"] += 1
        
        end_time = time.time()
        
        # Store Summary
        global_results[name] = {
            "accuracy": (correct / total) * 100 if total > 0 else 0,
            "total_samples": total,
            "latency_ms": ((end_time - start_time) / total) * 1000 if total > 0 else 0,
            "per_class": per_class
        }

    # Final Report Printing
    print("\n" + "="*60)
    print(f"{'Model':<15} | {'Accuracy':<10} | {'Latency (ms)':<12}")
    print("-" * 45)
    for name, res in global_results.items():
        print(f"{name:<15} | {res['accuracy']:>8.2f}% | {res['latency_ms']:>10.2f}")
    print("="*60)
    
    return global_results

if __name__ == "__main__":
    evaluate_all()
