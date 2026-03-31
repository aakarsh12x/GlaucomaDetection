import os
import random
from ultralytics import YOLO

def test_glaucoma_model():
    weights_path = os.path.join("runs", "classify", "glaucoma_runs", "yolo11_glaucoma_classification2", "weights", "best.pt")
    
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        return

    print(f"Loading trained model: {weights_path}")
    model = YOLO(weights_path)

    val_dir = os.path.join("yolo_dataset", "val")
    classes = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    
    print("\n--- COMPREHENSIVE MODEL EVALUATION ---")
    
    total_samples = 0
    total_correct = 0
    results_summary = {cls: {"correct": 0, "total": 0} for cls in classes}
    
    for cls in classes:
        cls_dir = os.path.join(val_dir, cls)
        images = [img for img in os.listdir(cls_dir) if img.endswith(".jpg") or img.endswith(".png")]
        
        if not images:
            continue
            
        # Test up to 25 samples per class for a balanced view
        samples = random.sample(images, min(len(images), 25))
        
        for img_name in samples:
            test_img = os.path.join(cls_dir, img_name)
            results = model.predict(test_img, imgsz=224, verbose=False)
            
            probs = results[0].probs
            names = results[0].names
            
            predicted_class = names[probs.top1]
            
            total_samples += 1
            results_summary[cls]["total"] += 1
            
            if predicted_class == cls:
                total_correct += 1
                results_summary[cls]["correct"] += 1

    print("\nFinal Report Summary:")
    print(f"Total Samples Tested: {total_samples}")
    print(f"Overall Accuracy: {(total_correct / total_samples) * 100:.2f}%")
    
    print("\nPer-Class Breakdown:")
    for cls, stats in results_summary.items():
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f" - {cls}: {acc:.2f}% ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    test_glaucoma_model()
