"""
Fine-tune ResNet18 on English Blueprint Dataset
Uses Voxel51/FloorPlanCAD (200 English floor plans)
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path

# Configuration
DATA_DIR = "english_data/floorplan_cad/images"
MODEL_OUTPUT = "blueprint_classifier_english.pth"
NUM_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 0.0001

class BlueprintDataset(Dataset):
    """Simple dataset for English floor plans"""
    def __init__(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.images = list(self.img_dir.glob("*.jpg"))
        self.transform = transform
        
        # Binary classification: blueprint vs non-blueprint
        # All images are blueprints (label=1)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # All are blueprints
        label = 1
        
        return image, label

def train_model():
    """Train ResNet18 on English blueprints"""
    
    print("="*60)
    print("Training ResNet18 on English Floor Plans")
    print("="*60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset
    dataset = BlueprintDataset(DATA_DIR, transform=transform)
    
    # Split: 80% train, 20% val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)  # 2 classes: blueprint/not blueprint
    )
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUTPUT)
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "="*60)
    print(f"✅ Training Complete!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved: {MODEL_OUTPUT}")
    print("="*60)
    
    return best_acc

if __name__ == "__main__":
    # Check data exists
    if not Path(DATA_DIR).exists():
        print(f"❌ Error: {DATA_DIR} not found!")
        print("Run: python download_english_floorplans.py first")
        exit(1)
    
    # Count images
    num_images = len(list(Path(DATA_DIR).glob("*.jpg")))
    print(f"Found {num_images} English floor plans")
    
    if num_images < 50:
        print("⚠ Warning: Less than 50 images. Consider downloading more data.")
    
    # Train
    best_acc = train_model()
