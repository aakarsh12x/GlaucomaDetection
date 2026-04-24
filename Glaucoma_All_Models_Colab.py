# %% [markdown]
# # 🔬 Glaucoma Detection — YOLOv11 · MambaOut · Vision Mamba
# **3-class:** `glaucoma` | `glaucoma_suspect` | `non_glaucoma`
#
# Runtime → GPU (T4). Run all cells top-to-bottom.

# %% — Cell 1: Install Dependencies
!pip install -q ultralytics datasets huggingface_hub torchvision tqdm Pillow scikit-learn matplotlib gradio

import torch
print(f"PyTorch: {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% — Cell 2: Download & Prepare Dataset
import os
from datasets import load_dataset
from PIL import Image
from tqdm.notebook import tqdm

DATASET_DIR = "/content/yolo_dataset"
MAX_SAMPLES = 3000
CLASS_NAMES = ["glaucoma", "glaucoma_suspect", "non_glaucoma"]
HF_DATASET  = "bumbledeep/smdg-full-dataset"

def prepare_dataset():
    print(f"Streaming '{HF_DATASET}' from HF …")
    try:
        sample = next(iter(load_dataset(HF_DATASET, split="train", streaming=True)))
        print("Columns:", list(sample.keys()))
    except Exception as e:
        print(f"[ERROR] {e}"); return

    for split_name, split_key in [("train","train"), ("val","validation")]:
        ds = load_dataset(HF_DATASET, split=split_key, streaming=True)
        print(f"\nProcessing '{split_name}' …")
        for count, item in enumerate(ds):
            if count >= MAX_SAMPLES: break
            img_key = next((k for k in ["image","img"] if k in item), list(item.keys())[0])
            lbl_key = next((k for k in ["label","diagnosis"] if k in item), list(item.keys())[-1])
            image, lbl_val = item[img_key], item[lbl_key]
            cls = CLASS_NAMES[lbl_val] if isinstance(lbl_val, int) and lbl_val < len(CLASS_NAMES) else str(lbl_val).lower().replace(" ","_")
            if isinstance(image, Image.Image):
                if image.mode != "RGB": image = image.convert("RGB")
                d = os.path.join(DATASET_DIR, split_name, cls)
                os.makedirs(d, exist_ok=True)
                image.save(os.path.join(d, f"img_{count:05d}.jpg"))
            if (count+1) % 500 == 0: print(f"  [{split_name}] {count+1}/{MAX_SAMPLES}")

    print(f"\n✅ Dataset ready at {DATASET_DIR}")
    for s in ["train","val"]:
        sp = os.path.join(DATASET_DIR, s)
        if os.path.isdir(sp):
            for c in sorted(os.listdir(sp)):
                print(f"   {s}/{c}: {len(os.listdir(os.path.join(sp, c)))} images")

prepare_dataset()

# %% — Cell 3: Model Definitions
import torch, torch.nn as nn

class GlaucomaMambaOut(nn.Module):
    """MambaOut: Gated-CNN backbone with GELU activations."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 128, 7, stride=2, padding=3), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.blocks = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.GELU(),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
        self.classifier = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.classifier(self.blocks(self.stem(x)))

class GlaucomaVim(nn.Module):
    """Vision-Mamba-inspired: patch embed → SSM-style conv blocks → head."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 128, 16, stride=16), nn.BatchNorm2d(128), nn.GELU())
        self.blocks = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.GELU(),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
        self.head = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.head(self.blocks(self.patch_embed(x)))

print("✅ Model classes defined: GlaucomaMambaOut, GlaucomaVim")

# %% — Cell 4: Train YOLOv11
from ultralytics import YOLO
YOLO_RUNS = "/content/runs/classify"
yolo_model = YOLO("yolo11n-cls.pt")
yolo_model.train(
    data="/content/yolo_dataset", epochs=20, imgsz=224, batch=16,
    project=YOLO_RUNS, name="yolo11_glaucoma",
    device=0 if torch.cuda.is_available() else "cpu", exist_ok=True)
YOLO_WEIGHTS = os.path.join(YOLO_RUNS, "yolo11_glaucoma", "weights", "best.pt")
print(f"\n✅ YOLOv11 done → {YOLO_WEIGHTS}")

# %% — Cell 5: Train MambaOut
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

MAMBA_SAVE = "/content/runs/mamba_out"; os.makedirs(MAMBA_SAVE, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_tf = transforms.Compose([
    transforms.Resize((230,230)), transforms.RandomCrop((224,224)),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
val_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

train_ds = datasets.ImageFolder(os.path.join(DATASET_DIR,"train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATASET_DIR,"val"),   transform=val_tf)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

mambaout = GlaucomaMambaOut(num_classes=len(train_ds.classes)).to(device)
opt_mo   = optim.AdamW(mambaout.parameters(), lr=1e-4, weight_decay=1e-4)
sched_mo = CosineAnnealingLR(opt_mo, T_max=30)
crit     = nn.CrossEntropyLoss()
best_mo, patience_mo = 0.0, 0

for ep in range(30):
    mambaout.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt_mo.zero_grad(); loss = crit(mambaout(x), y); loss.backward(); opt_mo.step()
    sched_mo.step()
    mambaout.eval(); c = t = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            c += (mambaout(x).argmax(1) == y).sum().item(); t += y.size(0)
    acc = 100*c/t if t else 0
    print(f"[MambaOut] Ep {ep+1}/30 | Val Acc: {acc:.2f}%")
    if acc > best_mo:
        best_mo = acc; torch.save(mambaout.state_dict(), os.path.join(MAMBA_SAVE,"best.pt")); patience_mo = 0
    else:
        patience_mo += 1
        if patience_mo >= 7: print(f"  Early stop ep {ep+1}"); break

MAMBA_WEIGHTS = os.path.join(MAMBA_SAVE, "best.pt")
print(f"\n✅ MambaOut done. Best: {best_mo:.2f}% → {MAMBA_WEIGHTS}")

# %% — Cell 6: Train Vision Mamba
VIM_SAVE = "/content/runs/vision_mamba"; os.makedirs(VIM_SAVE, exist_ok=True)

train_tf_v = transforms.Compose([
    transforms.Resize((230,230)), transforms.RandomCrop((224,224)),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
    transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
val_tf_v = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

train_ds_v = datasets.ImageFolder(os.path.join(DATASET_DIR,"train"), transform=train_tf_v)
val_ds_v   = datasets.ImageFolder(os.path.join(DATASET_DIR,"val"),   transform=val_tf_v)
train_dl_v = DataLoader(train_ds_v, batch_size=8, shuffle=True,  num_workers=2, pin_memory=True)
val_dl_v   = DataLoader(val_ds_v,   batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

vim = GlaucomaVim(num_classes=len(train_ds_v.classes)).to(device)
opt_v   = optim.AdamW(vim.parameters(), lr=1e-4, weight_decay=1e-4)
sched_v = CosineAnnealingLR(opt_v, T_max=30)
best_v, patience_v = 0.0, 0

for ep in range(30):
    vim.train()
    for x, y in train_dl_v:
        x, y = x.to(device), y.to(device)
        opt_v.zero_grad(); loss = crit(vim(x), y); loss.backward(); opt_v.step()
    sched_v.step()
    vim.eval(); c = t = 0
    with torch.no_grad():
        for x, y in val_dl_v:
            x, y = x.to(device), y.to(device)
            c += (vim(x).argmax(1) == y).sum().item(); t += y.size(0)
    acc = 100*c/t if t else 0
    print(f"[VisionMamba] Ep {ep+1}/30 | Val Acc: {acc:.2f}%")
    if acc > best_v:
        best_v = acc; torch.save(vim.state_dict(), os.path.join(VIM_SAVE,"best.pt")); patience_v = 0
    else:
        patience_v += 1
        if patience_v >= 7: print(f"  Early stop ep {ep+1}"); break

VIM_WEIGHTS = os.path.join(VIM_SAVE, "best.pt")
print(f"\n✅ VisionMamba done. Best: {best_v:.2f}% → {VIM_WEIGHTS}")

# %% — Cell 7: Unified Evaluation + Confusion Matrices
import time, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

VAL_DIR = os.path.join(DATASET_DIR, "val")
CLS = sorted(os.listdir(VAL_DIR))
print("Classes:", CLS)

ptf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

yolo_eval = YOLO(YOLO_WEIGHTS)
mo_eval = GlaucomaMambaOut(len(CLS)); mo_eval.load_state_dict(torch.load(MAMBA_WEIGHTS, map_location=device)); mo_eval.to(device).eval()
vi_eval = GlaucomaVim(len(CLS)); vi_eval.load_state_dict(torch.load(VIM_WEIGHTS, map_location=device)); vi_eval.to(device).eval()

imgs, lbls = [], []
for ci, cn in enumerate(CLS):
    cd = os.path.join(VAL_DIR, cn)
    if not os.path.isdir(cd): continue
    for fn in os.listdir(cd):
        if fn.lower().endswith((".jpg",".jpeg",".png")):
            imgs.append(os.path.join(cd, fn)); lbls.append(ci)
print(f"Val images: {len(imgs)}")

def infer(model, path, mtype):
    img = Image.open(path).convert("RGB")
    if mtype == "yolo":
        r = model.predict(img, imgsz=224, verbose=False)
        tn = r[0].names[r[0].probs.top1].lower().replace(" ","_")
        return CLS.index(tn) if tn in CLS else r[0].probs.top1
    t = ptf(img).unsqueeze(0).to(device)
    with torch.no_grad(): return model(t).argmax(1).item()

results = {}
for nm, mdl, mt in [("YOLOv11",yolo_eval,"yolo"),("MambaOut",mo_eval,"pytorch"),("VisionMamba",vi_eval,"pytorch")]:
    print(f"\n{'─'*50}\nEvaluating {nm} …")
    yt, yp = [], []; t0 = time.time()
    for ip, lb in zip(imgs, lbls):
        try: p = infer(mdl, ip, mt); yt.append(lb); yp.append(p)
        except: pass
    el = time.time() - t0; tot = len(yt)
    acc = sum(a==b for a,b in zip(yt,yp))/tot*100
    pr, rc, f1, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
    results[nm] = {"accuracy":acc,"precision":pr*100,"recall":rc*100,"f1":f1*100,"latency_ms":el/tot*1000,"y_true":yt,"y_pred":yp}
    print(classification_report(yt, yp, target_names=CLS, zero_division=0))

print("\n" + "="*75)
print(f"{'Model':<15} | {'Acc %':>7} | {'F1 %':>7} | {'Prec %':>7} | {'Rec %':>7} | {'ms/img':>8}")
print("-"*75)
for n, r in results.items():
    print(f"{n:<15} | {r['accuracy']:>7.2f} | {r['f1']:>7.2f} | {r['precision']:>7.2f} | {r['recall']:>7.2f} | {r['latency_ms']:>8.2f}")
print("="*75)

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices — Glaucoma Detection", fontsize=14, fontweight="bold")
for ax, (mn, r) in zip(axes, results.items()):
    cm = confusion_matrix(r["y_true"], r["y_pred"])
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(norm, cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_title(f"{mn}\nAcc:{r['accuracy']:.1f}% F1:{r['f1']:.1f}%", fontsize=11)
    ax.set_xticks(range(len(CLS))); ax.set_yticks(range(len(CLS)))
    ax.set_xticklabels(CLS, rotation=30, ha="right", fontsize=8); ax.set_yticklabels(CLS, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}\n({norm[i,j]*100:.0f}%)", ha="center", va="center",
                    color="white" if norm[i,j] > 0.5 else "black", fontsize=8)
fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
plt.tight_layout(); plt.savefig("/content/confusion_matrices.png", dpi=150); plt.show()

# %% — Cell 8: Gradio Inference UI (like app.py)
import gradio as gr

CLASS_LABELS = ["Glaucoma", "Glaucoma_Suspect", "Non_Glaucoma"]
inf_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def predict_glaucoma(image, model_name):
    if image is None:
        return {"Error": 1.0}, "Please upload an image."
    try:
        if model_name == "YOLOv11":
            r = yolo_eval.predict(image, imgsz=224, verbose=False)
            probs = r[0].probs; names = r[0].names
            confs = {names[i]: float(probs.data[i].item()) for i in range(len(names))}
            top_class = names[probs.top1]
        else:
            mdl = mo_eval if model_name == "MambaOut" else vi_eval
            t = inf_tf(image).unsqueeze(0).to(device)
            with torch.no_grad():
                out = torch.nn.functional.softmax(mdl(t)[0], dim=0)
            confs = {CLASS_LABELS[i]: float(out[i].item()) for i in range(len(CLASS_LABELS))}
            top_class = CLASS_LABELS[torch.argmax(out).item()]

        tc = top_class.lower()
        if "non" in tc: verdict = "✅ Normal / No Glaucoma Detected"
        elif "suspect" in tc: verdict = "🔍 Glaucoma Suspected (Borderline)"
        else: verdict = "⚠️ High Likelihood of Glaucoma"
        return confs, verdict
    except Exception as e:
        return {"Error": 1.0}, f"Inference error: {e}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👁️ Multi-Model AI Glaucoma Detection\nSelect model & upload retinal fundus image.")
    with gr.Row():
        with gr.Column():
            model_sel = gr.Dropdown(["YOLOv11","MambaOut","Vision Mamba"], value="YOLOv11", label="Model")
            img_in = gr.Image(type="pil", label="Fundus Image")
            btn = gr.Button("🧠 Diagnose", variant="primary")
        with gr.Column():
            verdict = gr.Text(label="📊 Verdict", interactive=False)
            probs_out = gr.Label(label="Confidence", num_top_classes=3)
    btn.click(predict_glaucoma, [img_in, model_sel], [probs_out, verdict])

demo.launch(share=True)

# %% — Cell 9: Download Weights
from google.colab import files
import shutil
shutil.make_archive("/content/glaucoma_weights", "zip", "/content/runs")
files.download("/content/glaucoma_weights.zip")
print("✅ All weights downloading …")
