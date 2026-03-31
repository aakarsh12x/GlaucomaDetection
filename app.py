import os
import gradio as gr
from PIL import Image
try:
    from ultralytics import YOLO
except ImportError:
    import sys
    sys.exit("Error: 'ultralytics' library is not installed. Please run: pip install -r requirements.txt")

# Define paths
# Our training script saves to a project directory named 'glaucoma_runs' at the execution root
TRAINED_WEIGHTS = os.path.join("runs", "classify", "glaucoma_runs", "yolo11_glaucoma_classification2", "weights", "best.pt")

print("Initializing YOLO Model...")
# Verify if the custom model weights exist from your training session
if os.path.exists(TRAINED_WEIGHTS):
    print(f"Loading custom trained YOLO weights from: {TRAINED_WEIGHTS}")
    model = YOLO(TRAINED_WEIGHTS)
else:
    print(f"WARNING: Trained weights not found at {TRAINED_WEIGHTS}.")
    print("Loading base yolo11n-cls.pt model. Please train your model using train_yolo11.py for actual diagnosis.")
    # Fallback to base model for demonstration
    model = YOLO("yolo11n-cls.pt")

def predict_glaucoma(image):
    """
    Runs YOLOv11 Classification on the input retina/fundus image.
    Format the results into a readable dictionary for Gradio.
    """
    if image is None:
        return "Please upload an image."

    # Perform inference
    results = model.predict(image, imgsz=224, verbose=False)

    # YOLO classification results contain a '.probs' attribute
    # with confidences for each class it was trained on.
    if hasattr(results[0], 'probs') and getattr(results[0], 'names', None) is not None:
        probs = results[0].probs
        names = results[0].names 
        
        # Get top 1 prediction
        top1_index = probs.top1
        top1_class = names[top1_index]
        confidence = probs.data[top1_index].item()
        
        # Create a dictionary of all class probabilities for output
        class_confidences = {names[i]: float(probs.data[i].item()) for i in range(len(names))}
        
        # Human-readable result
        diagnosis = "Glaucoma Detected" if "glaucoma" in top1_class.lower() else "Normal (No Glaucoma)"
        if top1_class not in ["normal", "glaucoma"]:
             # If using base Imagenet model, it might predict random objects.
             diagnosis = top1_class.capitalize()

        return class_confidences
    else:
        return {"Error": 1.0, "Message": "Model did not output classification probabilities."}

# Build Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 👁️ YOLOv11 AI Glaucoma Detection App\nUpload a retinal fundus image, and the YOLO classification model will instantly predict the presence of Glaucoma based on its training.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Patient Fundus Upload")
            predict_btn = gr.Button("Analyze Retinal Image", variant="primary")
            
        with gr.Column():
            prediction_output = gr.Label(label="Diagnosis Confidence & Classes", num_top_classes=3)
            
    # Connect input to processing function to output
    predict_btn.click(fn=predict_glaucoma, inputs=input_image, outputs=prediction_output)
    
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True)
