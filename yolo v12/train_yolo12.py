from ultralytics import YOLO

def train_glaucoma_yolo12():
    """
    Trains a YOLOv12 model to perform binary classification (Normal vs Glaucoma) on fundus images.
    YOLOv12 improves upon prior iterations with an Attention-Centric Real-Time object detection architecture.
    """
    print("Initializing YOLOv12 Architecture for Glaucoma Detection...")

    # Load the new YOLOv12 Nano classification model (using the standard ultralytics format convention)
    # Be sure to have the latest ultralytics package installed that supports yolov12 weights
    model = YOLO('yolov12n-cls.pt')

    # Data Structure required:
    # dataset/
    # ├── train/
    # │   ├── Normal/
    # │   └── Glaucoma/
    # └── val/
    #     ├── Normal/
    #     └── Glaucoma/

    print("YOLOv12 script is ready. Provide the proper dataset path to start training.")
    
    # UNCOMMENT to execute training on the Glaucoma dataset
    # results = model.train(
    #     data='path/to/glaucoma_classification_dataset',
    #     epochs=50,
    #     imgsz=224,       # Standard dimension for classification heads
    #     batch=16,
    #     name='yolo_v12_glaucoma_classification'
    # )
    
    # Example inference:
    # results = model.predict("path/to/fundus_test.jpg")
    # print(results)

if __name__ == '__main__':
    train_glaucoma_yolo12()
