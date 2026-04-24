import os
import sys
import logging
from typing import Tuple, Dict, Any, Optional
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure models can be imported from the models/ directory
sys.path.append(os.path.join(os.getcwd(), "models"))

try:
    from ultralytics import YOLO
except ImportError:
    logging.critical("ultralytics library not found. Run pip install ultralytics.")
    sys.exit("Error: 'ultralytics' library not found. Run pip install ultralytics.")

from mamba_out.train import GlaucomaMambaOut
from vision_mamba.train import GlaucomaVim

# Centralized Weights Map
WEIGHTS_MAP: Dict[str, str] = {
    "YOLOv11": os.path.join("runs", "classify", "glaucoma_runs", "yolo11_glaucoma_classification", "weights", "best.pt"),
    "MambaOut": os.path.join("runs", "mamba_out", "best.pt"),
    "Vision Mamba": os.path.join("runs", "vision_mamba", "best.pt")
}

CLASS_NAMES_MAMBA = ["Glaucoma", "Glaucoma_Suspect", "Non_Glaucoma"]

# Lazy-loaded model dictionary
loaded_models: Dict[str, Any] = {}

def get_model(model_name: str) -> Optional[Any]:
    """ Loads model architecture and weights on demand to conserve memory. """
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    path = WEIGHTS_MAP.get(model_name)
    if not path or not os.path.exists(path):
        logging.error(f"Weights for {model_name} not found at {path}")
        return None
    
    logging.info(f"Loading {model_name} model from {path}...")
    
    try:
        if "YOLO" in model_name:
            model = YOLO(path)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if model_name == "MambaOut":
                model = GlaucomaMambaOut(num_classes=3)
            else:
                model = GlaucomaVim(num_classes=3)
                
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
            
        loaded_models[model_name] = model
        return model
    except Exception as e:
        logging.error(f"Failed to load {model_name}: {e}")
        return None

# PyTorch Preprocessing Pipeline
pytorch_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_glaucoma(image: Optional[Image.Image], model_name: str) -> Tuple[Dict[str, float], str]:
    """
    General prediction function supporting both Ultralytics and native PyTorch models.
    """
    if image is None:
        return {"Error": 1.0}, "Please upload an image."

    model = get_model(model_name)
    if model is None:
        return {"Error": 1.0}, f"Error: Model weights for {model_name} missing or failed to load."

    try:
        if "YOLO" in model_name:
            results = model.predict(image, imgsz=224, verbose=False)
            probs = results[0].probs
            names = results[0].names
            
            confidences = {names[i]: float(probs.data[i].item()) for i in range(len(names))}
            top_idx = probs.top1
            top_class = names[top_idx]
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            img_tensor = pytorch_transforms(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                
            confidences = {CLASS_NAMES_MAMBA[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES_MAMBA))}
            top_idx = torch.argmax(probs).item()
            top_class = CLASS_NAMES_MAMBA[top_idx]

        t_low = top_class.lower()
        if "non" in t_low or "normal" in t_low:
            verdict = "✅ Normal / No Glaucoma Detected"
        elif "suspect" in t_low:
            verdict = "🔍 Glaucoma Suspected (Borderline Case)"
        elif "glaucoma" in t_low:
            verdict = "⚠️ High Likelihood of Glaucoma"
        else:
            verdict = f"Diagnosis: {top_class.capitalize()}"

        return confidences, verdict
    except Exception as e:
        logging.error(f"Inference error with {model_name}: {e}")
        return {"Error": 1.0}, "An error occurred during inference. Check logs."

# --- Gradio UI Setup ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")) as demo:
    gr.Markdown("""
    # 👁️ Multi-Model AI Glaucoma Detection
    Select a model architecture and upload a retinal fundus image for diagnosis. 
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=["YOLOv11", "MambaOut", "Vision Mamba"],
                value="YOLOv11",
                label="Architecture Selection"
            )
            input_image = gr.Image(type="pil", label="Patient Fundus Upload (Retina)")
            predict_btn = gr.Button("🧠 Start Diagnosis", variant="primary")
            
        with gr.Column(scale=1):
            label_verdict = gr.Text(label="📊 Final Verdict", interactive=False)
            prediction_output = gr.Label(
                label="Diagnosis Confidence Probability Breakdown",
                num_top_classes=3
            )
            gr.Markdown("""
            ---
            **Note on Classes:**
            - **Glaucoma**: High indications of optic nerve damage.
            - **Glaucoma_Suspect**: Borderline findings, follow-up recommended.
            - **Non_Glaucoma**: Normal retinal presentations.
            """)
            
    predict_btn.click(
        fn=predict_glaucoma, 
        inputs=[input_image, model_selector], 
        outputs=[prediction_output, label_verdict]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True)
