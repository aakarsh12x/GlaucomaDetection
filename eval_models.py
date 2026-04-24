import os
import sys
import torch
import time
import argparse
import logging
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support
from ultralytics import YOLO

# Setup module-level logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Add models to sys.path
sys.path.append(os.path.join(os.getcwd(), "models"))
from mamba_out.train import GlaucomaMambaOut
from vision_mamba.train import GlaucomaVim

# Transforms for PyTorch
pytorch_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CLASS_MAP = ["glaucoma", "glaucoma_suspect", "non_glaucoma"]

def get_models_config(runs_dir="runs"):
    return {
        "YOLOv11": {
            "type": "ultralytics",
            "path": os.path.join(runs_dir, "classify", "glaucoma_runs", "yolo11_glaucoma_classification", "weights", "best.pt")
        },
        "MambaOut": {
            "type": "pytorch",
            "class": GlaucomaMambaOut,
            "path": os.path.join(runs_dir, "mamba_out", "best.pt")
        },
        "Vision Mamba": {
            "type": "pytorch",
            "class": GlaucomaVim,
            "path": os.path.join(runs_dir, "vision_mamba", "best.pt")
        }
    }

def evaluate_all(data_dir, device_mode, runs_dir):
    device = torch.device('cuda' if device_mode == 'cuda' and torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    config_map = get_models_config(runs_dir)
    global_results = {}

    for name, config in config_map.items():
        logging.info(f"Evaluating {name}...")
        
        if not os.path.exists(config["path"]):
            logging.warning(f"Skipping {name}: Weights not found at {config['path']}")
            continue

        # Load Model
        try:
            if config["type"] == "ultralytics":
                model = YOLO(config["path"])
            else:
                model = config["class"](num_classes=len(CLASS_MAP))
                model.load_state_dict(torch.load(config["path"], map_location=device))
                model.to(device)
                model.eval()
        except Exception as e:
            logging.error(f"Failed to load {name}: {e}")
            continue

        # Run Eval Tracking
        y_true = []
        y_pred = []
        total = 0
        
        start_time = time.time()
        
        for class_idx, cls in enumerate(CLASS_MAP):
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                logging.warning(f"Class directory missing: {cls_dir}")
                continue
            
            images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in tqdm(images, desc=f"{name} - {cls}"):
                img_path = os.path.join(cls_dir, img_name)
                
                try:
                    img = Image.open(img_path).convert('RGB')
                except Exception as e:
                    logging.warning(f"Failed to read image {img_path}: {e}")
                    continue
                
                # Inference
                if config["type"] == "ultralytics":
                    results = model.predict(img, imgsz=224, verbose=False)
                    top_idx = results[0].probs.top1
                    pred_name = results[0].names[top_idx].lower().replace(" ", "_")
                    
                    if pred_name in CLASS_MAP:
                        pred_idx = CLASS_MAP.index(pred_name)
                    else:
                        pred_idx = top_idx
                else:
                    img_tensor = pytorch_transforms(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        pred_idx = torch.argmax(outputs[0]).item()
                
                y_true.append(class_idx)
                y_pred.append(pred_idx)
                total += 1
        
        end_time = time.time()
        
        if total == 0:
            logging.warning(f"No valid images processed for {name}.")
            continue

        # Metrics calculation
        accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / total * 100
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        global_results[name] = {
            "accuracy": accuracy,
            "f1_score": f1 * 100,
            "precision": precision * 100,
            "recall": recall * 100,
            "latency_ms": ((end_time - start_time) / total) * 1000
        }

    # Final Report Printing
    print("\n" + "="*90)
    print(f"{'Model':<15} | {'Acc (%)':<8} | {'F1 (%)':<8} | {'Prec (%)':<8} | {'Rec (%)':<8} | {'Latency(ms)':<10}")
    print("-" * 90)
    for name, res in global_results.items():
        print(f"{name:<15} | {res['accuracy']:>8.2f} | {res['f1_score']:>8.2f} | {res['precision']:>8.2f} | {res['recall']:>8.2f} | {res['latency_ms']:>10.2f}")
    print("="*90)
    
    return global_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Glaucoma Classification Models")
    parser.add_argument("--data_dir", type=str, default="yolo_dataset/val", help="Path to evaluation dataset")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Path to weights runs directory")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default="auto", help="Compute device to use")
    
    args = parser.parse_args()
    
    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        
    evaluate_all(args.data_dir, dev, args.runs_dir)
