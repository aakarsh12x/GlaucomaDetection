
# ============================================================
#  GLAUCOMA DETECTION — ALL 3 MODELS — GOOGLE COLAB NOTEBOOK
#  Copy each cell block into a separate Colab cell.
#  Runtime -> Change Runtime Type -> GPU (T4 recommended)
# ============================================================

# %% [markdown]
# # 🔬 Glaucoma Detection — YOLOv11 · MambaOut · Vision Mamba
# **3-class classification:** `glaucoma` | `glaucoma_suspect` | `non_glaucoma`
#
# **Instructions:**
# 1. Set Runtime → GPU (T4 or better)
# 2. Run all cells top-to-bottom
# 3. Models are saved to `/content/runs/` after training
# 4. Final cell runs unified evaluation across all 3 models

# %% [markdown]
# ## Cell 1 — Install Dependencies

# %%
# ─── Cell 1: Install Dependencies ───────────────────────────────────────────
!pip install -q ultralytics datasets huggingface_hub torchvision tqdm Pillow scikit-learn matplotlib

import torch
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA OK  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Cell 2 — Download & Prepare Dataset (Hugging Face Streaming)

# %%
# ─── Cell 2: Download Dataset ────────────────────────────────────────────────
import os
from datasets import load_dataset
from PIL import Image
from tqdm.notebook import tqdm

DATASET_DIR  = "/content/yolo_dataset"
MAX_SAMPLES  = 3000   # per split — increase if you want more data / better accuracy

CLASS_NAMES  = ["glaucoma", "glaucoma_suspect", "non_glaucoma"]   # target folder names
HF_DATASET   = "bumbledeep/smdg-full-dataset"

def prepare_dataset(download_dir=DATASET_DIR, max_per_split=MAX_SAMPLES):
    print(f"Streaming dataset '{HF_DATASET}' from Hugging Face …")

    try:
        ds_train = load_dataset(HF_DATASET, split="train",      streaming=True)
        ds_val   = load_dataset(HF_DATASET, split="validation", streaming=True)
    except Exception as e:
        print(f"[ERROR] Could not load dataset: {e}")
        return

    # Inspect first item so we know the column names
    sample = next(iter(ds_train))
    print("Dataset columns:", list(sample.keys()))

    # Re-create iterators (consumed above)
    ds_train = load_dataset(HF_DATASET, split="train",      streaming=True)
    ds_val   = load_dataset(HF_DATASET, split="validation", streaming=True)

    splits = {"train": ds_train, "val": ds_val}

    for split_name, ds_iter in splits.items():
        print(f"\nProcessing '{split_name}' split …")
        count = 0
        for item in ds_iter:
            if count >= max_per_split:
                break

            # ── Image ──
            img_key = next((k for k in ["image", "img"] if k in item), list(item.keys())[0])
            image   = item[img_key]

            # ── Label ──
            lbl_key = next((k for k in ["label", "diagnosis"] if k in item), list(item.keys())[-1])
            lbl_val = item[lbl_key]

            # Map label → class folder
            if isinstance(lbl_val, int) and lbl_val < len(CLASS_NAMES):
                class_name = CLASS_NAMES[lbl_val]
            else:
                # fallback — use string as folder name (clean it)
                class_name = str(lbl_val).lower().replace(" ", "_")

            # ── Save ──
            if isinstance(image, Image.Image):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                save_dir = os.path.join(download_dir, split_name, class_name)
                os.makedirs(save_dir, exist_ok=True)
                image.save(os.path.join(save_dir, f"img_{count:05d}.jpg"))

            count += 1
            if count % 200 == 0:
                print(f"  [{split_name}] {count}/{max_per_split}")

    print(f"\n✅ Dataset ready at {download_dir}")
    for split in ["train", "val"]:
        split_path = os.path.join(download_dir, split)
        if os.path.isdir(split_path):
            for cls in os.listdir(split_path):
                n = len(os.listdir(os.path.join(split_path, cls)))
                print(f"   {split}/{cls}: {n} images")


prepare_dataset()

# %% [markdown]
# ## Cell 3 — Model Definitions

# %%
# ─── Cell 3: Model Definitions ───────────────────────────────────────────────
import torch
import torch.nn as nn

# ── MambaOut ──────────────────────────────────────────────────────────────────
class GlaucomaMambaOut(nn.Module):
    """
    Inspired by MambaOut: Gated-CNN backbone with GELU activations.
    Stem → stacked gated conv blocks → classifier head.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        print(f"[MambaOut] Initialising — {num_classes} classes")

        self.stem = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.classifier(self.blocks(self.stem(x)))


# ── Vision Mamba ──────────────────────────────────────────────────────────────
class GlaucomaVim(nn.Module):
    """
    Vision-Mamba-inspired architecture: patch embed → SSM-style conv blocks → head.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        print(f"[VisionMamba] Initialising — {num_classes} classes")

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=16, stride=16),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.head(self.blocks(self.patch_embed(x)))


print("✅ Model classes defined: GlaucomaMambaOut, GlaucomaVim")

# %% [markdown]
# ## Cell 4 — Train YOLOv11

# %%
# ─── Cell 4: Train YOLOv11 ───────────────────────────────────────────────────
from ultralytics import YOLO
import os

DATASET_DIR = "/content/yolo_dataset"
YOLO_RUNS   = "/content/runs/classify"

yolo_model = YOLO("yolo11n-cls.pt")   # downloads ~5 MB nano classification checkpoint

results_yolo = yolo_model.train(
    data    = DATASET_DIR,
    epochs  = 20,           # increase to 30-50 for better accuracy
    imgsz   = 224,
    batch   = 16,
    project = YOLO_RUNS,
    name    = "yolo11_glaucoma",
    device  = 0 if torch.cuda.is_available() else "cpu",
    exist_ok= True,
)

YOLO_WEIGHTS = os.path.join(YOLO_RUNS, "yolo11_glaucoma", "weights", "best.pt")
print(f"\n✅ YOLOv11 training done. Weights → {YOLO_WEIGHTS}")

# %% [markdown]
# ## Cell 5 — Train MambaOut

# %%
# ─── Cell 5: Train MambaOut ──────────────────────────────────────────────────
import os, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

DATASET_DIR  = "/content/yolo_dataset"
MAMBA_SAVE   = "/content/runs/mamba_out"
EPOCHS       = 30
BATCH_SIZE   = 16
LR           = 1e-4
PATIENCE     = 7

os.makedirs(MAMBA_SAVE, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_tf = transforms.Compose([
    transforms.Resize((230, 230)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"),   transform=val_tf)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

num_classes   = len(train_ds.classes)
mambaout_mdl  = GlaucomaMambaOut(num_classes=num_classes).to(device)
optimizer_mo  = optim.AdamW(mambaout_mdl.parameters(), lr=LR, weight_decay=1e-4)
scheduler_mo  = CosineAnnealingLR(optimizer_mo, T_max=EPOCHS)
criterion_mo  = nn.CrossEntropyLoss()

best_acc_mo   = 0.0
no_improve_mo = 0

for epoch in range(EPOCHS):
    # ── Train ──
    mambaout_mdl.train()
    for inputs, labels in train_dl:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_mo.zero_grad()
        loss = criterion_mo(mambaout_mdl(inputs), labels)
        loss.backward()
        optimizer_mo.step()
    scheduler_mo.step()

    # ── Validate ──
    mambaout_mdl.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in val_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            preds  = mambaout_mdl(inputs).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    val_acc = 100 * correct / total if total else 0
    print(f"[MambaOut] Epoch {epoch+1:02d}/{EPOCHS} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc_mo:
        best_acc_mo = val_acc
        torch.save(mambaout_mdl.state_dict(), os.path.join(MAMBA_SAVE, "best.pt"))
        no_improve_mo = 0
        print(f"  ↑ New best saved ({best_acc_mo:.2f}%)")
    else:
        no_improve_mo += 1
        if no_improve_mo >= PATIENCE:
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break

MAMBA_WEIGHTS = os.path.join(MAMBA_SAVE, "best.pt")
print(f"\n✅ MambaOut training done. Best Val Acc: {best_acc_mo:.2f}% → {MAMBA_WEIGHTS}")

# %% [markdown]
# ## Cell 6 — Train Vision Mamba

# %%
# ─── Cell 6: Train Vision Mamba ──────────────────────────────────────────────
import os, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

DATASET_DIR = "/content/yolo_dataset"
VIM_SAVE    = "/content/runs/vision_mamba"
EPOCHS_VIM  = 30
BATCH_VIM   = 8
LR_VIM      = 1e-4
PATIENCE_VIM= 7

os.makedirs(VIM_SAVE, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_tf_vim = transforms.Compose([
    transforms.Resize((230, 230)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf_vim = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds_vim = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_tf_vim)
val_ds_vim   = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"),   transform=val_tf_vim)
train_dl_vim = DataLoader(train_ds_vim, batch_size=BATCH_VIM, shuffle=True,  num_workers=2, pin_memory=True)
val_dl_vim   = DataLoader(val_ds_vim,   batch_size=BATCH_VIM, shuffle=False, num_workers=2, pin_memory=True)

num_classes_vim = len(train_ds_vim.classes)
vim_mdl         = GlaucomaVim(num_classes=num_classes_vim).to(device)
optimizer_vim   = optim.AdamW(vim_mdl.parameters(), lr=LR_VIM, weight_decay=1e-4)
scheduler_vim   = CosineAnnealingLR(optimizer_vim, T_max=EPOCHS_VIM)
criterion_vim   = nn.CrossEntropyLoss()

best_acc_vim   = 0.0
no_improve_vim = 0

for epoch in range(EPOCHS_VIM):
    # ── Train ──
    vim_mdl.train()
    for inputs, labels in train_dl_vim:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_vim.zero_grad()
        loss = criterion_vim(vim_mdl(inputs), labels)
        loss.backward()
        optimizer_vim.step()
    scheduler_vim.step()

    # ── Validate ──
    vim_mdl.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in val_dl_vim:
            inputs, labels = inputs.to(device), labels.to(device)
            preds   = vim_mdl(inputs).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    val_acc = 100 * correct / total if total else 0
    print(f"[VisionMamba] Epoch {epoch+1:02d}/{EPOCHS_VIM} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc_vim:
        best_acc_vim = val_acc
        torch.save(vim_mdl.state_dict(), os.path.join(VIM_SAVE, "best.pt"))
        no_improve_vim = 0
        print(f"  ↑ New best saved ({best_acc_vim:.2f}%)")
    else:
        no_improve_vim += 1
        if no_improve_vim >= PATIENCE_VIM:
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break

VIM_WEIGHTS = os.path.join(VIM_SAVE, "best.pt")
print(f"\n✅ Vision Mamba training done. Best Val Acc: {best_acc_vim:.2f}% → {VIM_WEIGHTS}")

# %% [markdown]
# ## Cell 7 — Unified Evaluation (All 3 Models)

# %%
# ─── Cell 7: Unified Evaluation ──────────────────────────────────────────────
import os, torch, time
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_fscore_support)
import matplotlib.pyplot as plt
import numpy as np

DATASET_DIR = "/content/yolo_dataset"
VAL_DIR     = os.path.join(DATASET_DIR, "val")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = sorted(os.listdir(VAL_DIR))   # actual folder names
print("Classes:", CLASS_NAMES)

pytorch_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Load weights ─────────────────────────────────────────────────────────────
YOLO_WEIGHTS  = "/content/runs/classify/yolo11_glaucoma/weights/best.pt"
MAMBA_WEIGHTS = "/content/runs/mamba_out/best.pt"
VIM_WEIGHTS   = "/content/runs/vision_mamba/best.pt"

yolo_eval = YOLO(YOLO_WEIGHTS)

n_cls = len(CLASS_NAMES)
mambaout_eval = GlaucomaMambaOut(num_classes=n_cls)
mambaout_eval.load_state_dict(torch.load(MAMBA_WEIGHTS, map_location=DEVICE))
mambaout_eval.to(DEVICE).eval()

vim_eval = GlaucomaVim(num_classes=n_cls)
vim_eval.load_state_dict(torch.load(VIM_WEIGHTS, map_location=DEVICE))
vim_eval.to(DEVICE).eval()

# ── Gather all images ────────────────────────────────────────────────────────
all_images, all_labels = [], []
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    cls_dir = os.path.join(VAL_DIR, cls_name)
    if not os.path.isdir(cls_dir):
        continue
    for fn in os.listdir(cls_dir):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            all_images.append(os.path.join(cls_dir, fn))
            all_labels.append(cls_idx)

print(f"Total validation images: {len(all_images)}")

# ── Inference helper ──────────────────────────────────────────────────────────
def run_inference(model, img_path, model_type="pytorch"):
    img = Image.open(img_path).convert("RGB")
    if model_type == "yolo":
        res = model.predict(img, imgsz=224, verbose=False)
        top = res[0].probs.top1
        top_name = res[0].names[top].lower().replace(" ", "_")
        if top_name in CLASS_NAMES:
            return CLASS_NAMES.index(top_name)
        return top
    else:
        t = pytorch_tf(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return model(t).argmax(1).item()


# ── Evaluate each model ───────────────────────────────────────────────────────
model_configs = [
    ("YOLOv11",       yolo_eval,      "yolo"),
    ("MambaOut",      mambaout_eval,  "pytorch"),
    ("VisionMamba",   vim_eval,       "pytorch"),
]

results_table = {}

for model_name, model_obj, mtype in model_configs:
    print(f"\n{'─'*50}")
    print(f"Evaluating {model_name} …")
    y_true, y_pred = [], []
    t0 = time.time()

    for img_path, lbl in zip(all_images, all_labels):
        try:
            pred = run_inference(model_obj, img_path, mtype)
            y_true.append(lbl)
            y_pred.append(pred)
        except Exception as e:
            pass   # skip corrupt images

    elapsed = time.time() - t0
    total   = len(y_true)
    acc     = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / total * 100
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    results_table[model_name] = {
        "accuracy":    acc,
        "precision":   prec * 100,
        "recall":      rec  * 100,
        "f1":          f1   * 100,
        "latency_ms":  elapsed / total * 1000,
        "y_true":      y_true,
        "y_pred":      y_pred,
    }

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*75)
print(f"{'Model':<15} | {'Acc %':>7} | {'F1 %':>7} | {'Prec %':>7} | {'Rec %':>7} | {'ms/img':>8}")
print("-"*75)
for name, r in results_table.items():
    print(f"{name:<15} | {r['accuracy']:>7.2f} | {r['f1']:>7.2f} | {r['precision']:>7.2f} | {r['recall']:>7.2f} | {r['latency_ms']:>8.2f}")
print("="*75)

# %% [markdown]
# ## Cell 8 — Confusion Matrices

# %%
# ─── Cell 8: Confusion Matrices ──────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices — Glaucoma Detection", fontsize=14, fontweight="bold")

cmap = plt.cm.Blues

for ax, (model_name, r) in zip(axes, results_table.items()):
    cm   = confusion_matrix(r["y_true"], r["y_pred"])
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(f"{model_name}\nAcc: {r['accuracy']:.1f}%  F1: {r['f1']:.1f}%", fontsize=11)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}\n({norm[i,j]*100:.0f}%)",
                    ha="center", va="center",
                    color="white" if norm[i,j] > 0.5 else "black", fontsize=8)

fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("/content/confusion_matrices.png", dpi=150)
plt.show()
print("✅ Saved: /content/confusion_matrices.png")

# %% [markdown]
# ## Cell 9 — Download Trained Weights

# %%
# ─── Cell 9: Download Weights ────────────────────────────────────────────────
from google.colab import files
import shutil, os

ARCHIVE_PATH = "/content/glaucoma_weights.zip"
shutil.make_archive("/content/glaucoma_weights", "zip", "/content/runs")
files.download(ARCHIVE_PATH)
print("✅ All model weights packaged and downloading …")
# Contains:
#   runs/classify/yolo11_glaucoma/weights/best.pt   ← YOLOv11
#   runs/mamba_out/best.pt                           ← MambaOut
#   runs/vision_mamba/best.pt                        ← Vision Mamba
