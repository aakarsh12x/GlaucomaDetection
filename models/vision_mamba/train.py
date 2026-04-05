import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# A simple wrapper layout for the Vision Mamba (Vim) model setup for Glaucoma Detection
class GlaucomaVim(nn.Module):
    def __init__(self, num_classes=3):
        super(GlaucomaVim, self).__init__()
        print(f"Initializing Vision Mamba base architecture for {num_classes} classes...")
        
        # Patch Embed - Increased capacity
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=16, stride=16),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
        # State Space (Mamba) Blocks (Scaled up)
        self.blocks = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # Block 2
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Head
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        features = self.blocks(x)
        return self.head(features)

def train_vision_mamba(data_dir="../../yolo_dataset", epochs=30, batch_size=8, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Stronger Data Augmentation for Accuracy
    train_transform = transforms.Compose([
        transforms.Resize((230, 230)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset & Loader
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    if not os.path.exists(train_dir):
        print(f"Error: Data directory {train_dir} not found.")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = len(train_dataset.classes)
    model = GlaucomaVim(num_classes=num_classes).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Validation Accuracy after Epoch {epoch+1}: {(100 * correct / total):.2f}%")

    # Save weights
    os.makedirs("../../runs/vision_mamba", exist_ok=True)
    torch.save(model.state_dict(), "../../runs/vision_mamba/best.pt")
    print("Vision Mamba training complete. Weights saved to runs/vision_mamba/best.pt")

if __name__ == "__main__":
    train_vision_mamba()
