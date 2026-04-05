import os
import sys
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

# Ensure models can be imported from the models/ directory
sys.path.append(os.path.join(os.getcwd(), "models"))

try:
    from ultralytics import YOLO
except ImportError:
    import sys
    sys.exit("Error: 'ultralytics' library not found. Run pip install ultralytics.")

# Import custom model classes - these work because we added 'models' to sys.path
from mamba_out.train import GlaucomaMambaOut
from vision_mamba.train import GlaucomaVim

# Centralized Weights Map
WEIGHTS_MAP = {
    "YOLOv11": os.path.join("runs", "classify", "glaucoma_runs", "yolo11_glaucoma_classification", "weights", "best.pt"),
    "MambaOut": os.path.join("runs", "mamba_out", "best.pt"),
    "Vision Mamba": os.path.join("runs", "vision_mamba", "best.pt")
}

# Standard Class Names for Mamba models
# (YOLO models fetch names directly from the weights metadata)
CLASS_NAMES_MAMBA = ["Glaucoma", "Glaucoma_Suspect", "Non_Glaucoma"]

# Lazy-loaded model dictionary
loaded_models = {}

def get_model(model_name):
    """ Loads model architecture and weights on demand to conserve memory. """
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    path = WEIGHTS_MAP.get(model_name)
    if not os.path.exists(path):
        print(f"ERROR: Weights for {model_name} not found at {path}")
        return None
    
    print(f"Loading {model_name} model from {path}...")
    
    if "YOLO" in model_name:
        model = YOLO(path)
    else:
        # PyTorch Architectures
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

# PyTorch Preprocessing Pipeline
pytorch_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_glaucoma(image, model_name):
    """
    General prediction function supporting both Ultralytics and native PyTorch models.
    """
    if image is None:
        return "Please upload an image.", "No Input"

    model = get_model(model_name)
    if model is None:
        return {"Error": 1.0, f"Weights for {model_name} missing": 1.0}, "Error: Model not found"

    if "YOLO" in model_name:
        # YOLO Prediction
        results = model.predict(image, imgsz=224, verbose=False)
        probs = results[0].probs
        names = results[0].names
        
        # Format results for Gradio Label component
        confidences = {names[i]: float(probs.data[i].item()) for i in range(len(names))}
        top_idx = probs.top1
        top_class = names[top_idx]
    else:
        # PyTorch Prediction (Mamba/Vim)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_tensor = pytorch_transforms(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            
        confidences = {CLASS_NAMES_MAMBA[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES_MAMBA))}
        top_idx = torch.argmax(probs).item()
        top_class = CLASS_NAMES_MAMBA[top_idx]

    # Handle Verdict - Fix: check for "non" first to avoid string matching conflicts
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

# --- Gradio UI Setup ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")) as demo:
    gr.Markdown("""
    # 👁️ Multi-Model AI Glaucoma Detection
    Select a model architecture and upload a retinal fundus image for diagnosis. 
    """)
    
    with gr.Row():
        with gr.Column(scale=1): # Smaller width for inputs
            model_selector = gr.Dropdown(
                choices=["YOLOv11", "MambaOut", "Vision Mamba"],
                value="YOLOv11",
                label="Architecture Selection"
            )
            input_image = gr.Image(type="pil", label="Patient Fundus Upload (Retina)")
            predict_btn = gr.Button("🧠 Start Diagnosis", variant="primary")
            
        with gr.Column(scale=1): # Output column
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
    # Launch on localhost
    demo.launch(server_name="127.0.0.1", inbrowser=True)
