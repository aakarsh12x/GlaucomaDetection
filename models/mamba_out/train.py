import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# A simplified template representing MambaOut's approach for Glaucoma Detection
class GlaucomaMambaOut(nn.Module):
    def __init__(self, num_classes=3):
        super(GlaucomaMambaOut, self).__init__()
        print(f"Initializing MambaOut architecture for {num_classes} classes...")
        
        # Stem - Increased complexity
        self.stem = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # MambaOut-inspired blocks (Scaled up)
        self.blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # Block 2 (Residual-like)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # Block 3 (Reduction)
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        features = self.blocks(x)
        return self.classifier(features)

def train_mambaout(data_dir="../../yolo_dataset", epochs=30, batch_size=16, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Stronger Data Augmentation for Accuracy
    train_transform = transforms.Compose([
        transforms.Resize((230, 230)), # Resize slightly larger
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
    model = GlaucomaMambaOut(num_classes=num_classes).to(device)
    
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
    os.makedirs("../../runs/mamba_out", exist_ok=True)
    torch.save(model.state_dict(), "../../runs/mamba_out/best.pt")
    print("MambaOut training complete. Weights saved to runs/mamba_out/best.pt")

if __name__ == "__main__":
    train_mambaout()
