import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GlaucomaMambaOut(nn.Module):
    def __init__(self, num_classes=3):
        super(GlaucomaMambaOut, self).__init__()
        logging.info(f"Initializing MambaOut architecture for {num_classes} classes...")
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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
            nn.Flatten()
        )
        
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        features = self.blocks(x)
        return self.classifier(features)

def train_mambaout(data_dir="../../yolo_dataset", epochs=30, batch_size=16, lr=1e-4, patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Training on device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((230, 230)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    if not os.path.exists(train_dir):
        logging.error(f"Data directory {train_dir} not found.")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = len(train_dataset.classes)
    model = GlaucomaMambaOut(num_classes=num_classes).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    epochs_no_improve = 0
    save_dir = "../../runs/mamba_out"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best.pt")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i+1) % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        scheduler.step()
        
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total if total > 0 else 0
        logging.info(f"Validation Accuracy after Epoch {epoch+1}: {val_acc:.2f}%")

        # Checkpointing and Early Stopping
        if val_acc > best_val_acc:
            logging.info(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%. Saving model...")
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                logging.warning(f"Early stopping triggered after {epoch+1} epochs.")
                break

    logging.info(f"MambaOut training complete. Best Validation Accuracy: {best_val_acc:.2f}%. Weights saved to {best_model_path}")

if __name__ == "__main__":
    train_mambaout()
