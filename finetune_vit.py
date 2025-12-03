import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
import os
import glob
from PIL import Image

class RealFloorPlanDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.transform = transform
        # For this demo, we'll use unsupervised/self-supervised learning
        # Or create pseudo-labels based on image characteristics
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Create pseudo-label based on image characteristics
        # For real fine-tuning, you'd need actual labels
        # Here we'll use image statistics as a proxy
        img_array = np.array(img)
        mean_intensity = np.mean(img_array)
        
        # Simple categorization: dark/medium/light for demo
        if mean_intensity < 85:
            label = 0  # Dark/detailed floor plan
        elif mean_intensity < 170:
            label = 1  # Medium
        else:
            label = 2  # Light/simple floor plan
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def finetune_model(data_dir="real_data/images", epochs=5, batch_size=8):
    print("Setting up fine-tuning...")
    
    # Data augmentation for real floor plans
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),  # Small rotation for architectural drawings
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = RealFloorPlanDataset(data_dir, transform=transform)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # Load pre-trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 3)  # 3 classes for our pseudo-labels
    model = model.to(device)
    
    # Fine-tuning setup
    criterion = nn.CrossEntropyLoss()
    # Use lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Training loop
    best_val_loss = float('inf')
    print("\nStarting fine-tuning...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "blueprint_classifier_finetuned.pth")
            print("  → Saved improved model")
    
    print("\nFine-tuning complete!")
    print(f"Best model saved to blueprint_classifier_finetuned.pth")
    
    return model

if __name__ == "__main__":
    model = finetune_model()
