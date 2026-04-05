import os
import argparse
from ultralytics import YOLO

def train_glaucoma_yolo11(data_path="dataset", epochs=20, imgsz=224):
    """
    Trains a YOLOv11 model (yolo11n-cls.pt) to classify input datasets.
    Make sure your dataset_path points to the root folder holding train/ and val/ subdirectories.
    """
    print("Initializing YOLOv11 for Glaucoma Classification...")
    
    # Initialize the Nano Classification model for fast local training
    model = YOLO('yolo11n-cls.pt')

    if not os.path.exists(data_path):
        print(f"Error: Dataset path '{data_path}' not found.")
        print("Please run `python prepare_data.py` to download the public testing dataset.")
        return

    print(f"Starting Training on dataset localized at: {os.path.abspath(data_path)}")
    
    # Run YOLO Classification Training
    results = model.train(
        data=os.path.abspath(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        project='glaucoma_runs',
        name='yolo11_glaucoma_classification',
        device=0 # Use NVIDIA GPU (CUDA 0)
    )
    
    print("\nTraining complete! Your weights are saved inside the 'glaucoma_runs' directory.")
    print("You can evaluate inference using `app.py` UI script.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLOv11 for Glaucoma Classification.")
    parser.add_argument("--data", default="../../yolo_dataset", help="Path to your formatted dataset root.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for.")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size for training and validation.")
    
    args = parser.parse_args()
    
    train_glaucoma_yolo11(data_path=args.data, epochs=args.epochs, imgsz=args.imgsz)
