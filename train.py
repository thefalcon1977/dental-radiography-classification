"""
Dental Radiography Classification - Top Approach
Based on best practices for medical image classification
Optimized for Mac M4 with MPS support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# Set device - Mac M4 will use MPS (Metal Performance Shaders)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "mps":
    print("✅ MPS (Metal Performance Shaders) is available for GPU acceleration!")

# ============================================================================
# Dataset Class
# ============================================================================
class DentalRadiographyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Class mapping
        self.classes = ['dentin', 'enamel', 'pulp']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load images and labels
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in sorted(os.listdir(class_dir)):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.PNG')):
                        img_path = os.path.join(class_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# Data Transforms with Advanced Augmentation
# ============================================================================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# Load Datasets
# ============================================================================
train_dir = 'segmented_dental_adiography/train'
val_dir = 'segmented_dental_adiography/valid'
test_dir = 'segmented_dental_adiography/test'

print("\n" + "="*60)
print("Loading Datasets...")
print("="*60)

train_dataset = DentalRadiographyDataset(train_dir, transform=train_transform)
val_dataset = DentalRadiographyDataset(val_dir, transform=val_transform)
test_dataset = DentalRadiographyDataset(test_dir, transform=test_transform)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Check class distribution
train_labels = [train_dataset.labels[i] for i in range(len(train_dataset))]
class_counts = Counter(train_labels)
print("\nClass distribution in training set:")
for idx, count in sorted(class_counts.items()):
    print(f"  {train_dataset.idx_to_class[idx]}: {count} samples")

# ============================================================================
# Handle Class Imbalance with Weighted Sampler
# ============================================================================
class_weights = []
total_samples = len(train_labels)
for idx in range(3):
    class_count = class_counts.get(idx, 1)
    weight = total_samples / (3 * class_count)
    class_weights.append(weight)

sample_weights = [class_weights[label] for label in train_labels]
weighted_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# ============================================================================
# Data Loaders
# ============================================================================
batch_size = 16  # Reduced for Mac M4 memory
# Note: pin_memory is not supported on MPS, so we set it to False
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    sampler=weighted_sampler,
    num_workers=0,
    pin_memory=False
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=0,
    pin_memory=False
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=0,
    pin_memory=False
)

# ============================================================================
# Model Architecture - DenseNet121 (Top Choice for Medical Images)
# ============================================================================
def create_model(model_name='densenet121', num_classes=3, pretrained=True):
    """Create model with transfer learning"""
    if model_name == 'densenet121':
        if pretrained:
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            model = models.densenet121(weights=None)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    elif model_name == 'resnet50':
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == 'efficientnet_b0':
        if pretrained:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

# Create model
print("\n" + "="*60)
print("Creating Model...")
print("="*60)
model = create_model('densenet121', num_classes=3, pretrained=True)
model = model.to(device)
print(f"Model: DenseNet121")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================================
# Loss Function with Class Weights
# ============================================================================
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Optimizer with different learning rates for different layers
optimizer = optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': 1e-4},
    {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 'lr': 1e-3}
], weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

# ============================================================================
# Training Loop
# ============================================================================
num_epochs = 30
train_losses, train_accs = [], []
val_losses, val_accs = [], []

best_val_acc = 0.0
best_model_path = 'best_densenet_model.pth'
patience = 7
patience_counter = 0

print("\n" + "="*60)
print("Starting Training...")
print("="*60)

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 60)
    
    train_loss, train_acc, train_preds, train_labels = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc, val_preds, val_labels = validate(
        model, val_loader, criterion, device
    )
    
    # Get learning rate before scheduler step
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"Learning Rate: {new_lr:.6f}", end="")
    if new_lr < old_lr:
        print(f" (reduced from {old_lr:.6f})")
    else:
        print()
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, best_model_path)
        print(f"✅ Saved best model with validation accuracy: {val_acc:.2f}%")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# ============================================================================
# Load Best Model and Test
# ============================================================================
print("\n" + "="*60)
print("Testing on Test Set...")
print("="*60)

checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model from epoch {checkpoint['epoch']+1} with val_acc: {checkpoint['val_acc']:.2f}%")

test_loss, test_acc, test_preds, test_labels = validate(
    model, test_loader, criterion, device
)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")

# ============================================================================
# Detailed Evaluation Metrics
# ============================================================================
print("\n" + "="*60)
print("Detailed Classification Report")
print("="*60)

class_names = ['dentin', 'enamel', 'pulp']
print("\nTest Set Classification Report:")
print(classification_report(
    test_labels, 
    test_preds, 
    target_names=class_names,
    digits=4
))

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("\nConfusion matrix saved to confusion_matrix.png")

# ============================================================================
# Training History Visualization
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0].plot(val_losses, label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training and Validation Loss', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label='Train Acc', linewidth=2)
axes[1].plot(val_accs, label='Val Acc', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Training and Validation Accuracy', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
print("Training history saved to training_history.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("Training Summary")
print("="*60)
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Total Epochs Trained: {len(train_losses)}")
print(f"Best Model Saved: {best_model_path}")
print("="*60)

