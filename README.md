# Dental Radiography Classification with DenseNet

A deep learning system for classifying and detecting dental structures (dentin, enamel, pulp) in radiography images using DenseNet121, optimized for Apple Silicon (Mac M4) with MPS acceleration.

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Object Detection](#object-detection)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🔍 Overview

This project implements a state-of-the-art dental radiography classification system using transfer learning with DenseNet121. The system can:

- **Classify** dental structures into three categories: dentin, enamel, and pulp
- **Detect** and localize dental structures in full radiography images
- **Train** efficiently on Mac M4 using Metal Performance Shaders (MPS)
- **Evaluate** with comprehensive metrics and visualizations

### Key Capabilities

- ✅ High accuracy classification (98%+ validation accuracy)
- ✅ Real-time inference on Mac M4
- ✅ Object detection with bounding boxes
- ✅ Comprehensive evaluation metrics
- ✅ Easy-to-use command-line interface

## ✨ Features

### Model Architecture
- **DenseNet121** pretrained on ImageNet
- Transfer learning with fine-tuning
- Separate learning rates for backbone and classifier
- Optimized for medical image analysis

### Advanced Training Features
- **Data Augmentation**: Random crops, flips, rotations, color jittering, random erasing
- **Class Imbalance Handling**: Weighted sampling and weighted loss function
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Automatic stopping when validation stops improving
- **Model Checkpointing**: Saves best model based on validation accuracy

### Evaluation & Visualization
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Training history plots (loss and accuracy curves)
- Per-class accuracy metrics

### Hardware Acceleration
- **MPS Support**: Leverages Apple Silicon GPU acceleration
- **Optimized Batch Size**: Configured for Mac M4 memory
- **Efficient Data Loading**: Optimized for Apple Silicon architecture

## 📁 Dataset Structure

The dataset should be organized in the following structure:

```
segmented_dental_adiography/
├── train/
│   ├── dentin/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   ├── enamel/
│   │   └── ...
│   └── pulp/
│       └── ...
├── valid/
│   ├── dentin/
│   ├── enamel/
│   └── pulp/
└── test/
    ├── dentin/
    ├── enamel/
    └── pulp/
```

### Dataset Statistics
- **Training samples**: 563 images
- **Validation samples**: 54 images
- **Test samples**: 163 images
- **Classes**: 3 (dentin, enamel, pulp)
- **Class distribution**: Balanced (~189 samples per class in training)

## 🚀 Installation

### Prerequisites
- Python 3.13+
- macOS with Apple Silicon (M1/M2/M3/M4) for MPS acceleration
- 8GB+ RAM recommended

### Option 1: Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd densNet

# Run the setup script
./run_training.sh
```

This will:
1. Create a virtual environment
2. Install all dependencies
3. Start training automatically

### Option 2: Manual Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

## 🎯 Quick Start

### 1. Train the Model

```bash
# Activate virtual environment
source .venv/bin/activate

# Start training
python train.py
```

The training will:
- Use MPS (GPU) acceleration on Mac M4
- Train for up to 30 epochs with early stopping
- Save the best model to `best_densenet_model.pth`
- Generate training visualizations

### 2. Make Predictions

```bash
# Predict a single image
python detect_simple.py
# Then enter the image path when prompted
```

### 3. View Results

After training, you'll have:
- `best_densenet_model.pth` - Trained model weights
- `training_history.png` - Training/validation curves
- `confusion_matrix.png` - Test set confusion matrix

## 🎓 Training

### Basic Training

```bash
python train.py
```

### Training Parameters

The training script uses the following default parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 16 | Optimized for Mac M4 memory |
| Epochs | 30 | With early stopping (patience=7) |
| Learning Rate (backbone) | 1e-4 | For pretrained layers |
| Learning Rate (classifier) | 1e-3 | For new classification layer |
| Optimizer | AdamW | With weight decay 1e-4 |
| Image Size | 224×224 | Standard for DenseNet |
| Confidence Threshold | 0.5 | For detection tasks |

### Training Output

During training, you'll see:

```
Epoch 1/30
------------------------------------------------------------
Training: 100%|████████████████| 36/36 [00:10<00:00, 3.53it/s]
Validating: 100%|██████████████| 4/4 [00:00<00:00, 4.32it/s]
Train Loss: 1.0268, Train Acc: 49.38%
Val Loss: 0.7726, Val Acc: 68.52%
Learning Rate: 0.000100
✅ Saved best model with validation accuracy: 68.52%
```

### Training Features

**Data Augmentation (Training Set)**
- Resize to 256×256
- Random resized crop to 224×224 (scale 0.8-1.0)
- Random horizontal flip (50%)
- Random vertical flip (30%)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transformations
- Random erasing (20% probability)

**Validation/Test Transforms**
- Resize to 256×256
- Center crop to 224×224
- Normalization (ImageNet statistics)

### Monitoring Training

The script automatically:
- Prints progress for each epoch
- Shows learning rate changes
- Saves best model when validation improves
- Applies early stopping if no improvement for 7 epochs
- Generates training history plots

## 🔮 Inference

### Simple Prediction (Highest Confidence)

```bash
python detect_simple.py
```

**Features:**
- Interactive: prompts for image path
- Shows only the highest confidence detection
- Displays image with bounding box
- No file saving (display only)

**Output:**
```
Image size: 217x692
Scanning image...

✅ Highest confidence detection:
   Class: DENTIN
   Confidence: 99.56%
   Location: (0, 0) to (224, 224)

✅ Image displayed!
```

### Prediction Output

The displayed image shows:
- **Bounding box** around the detected region
- **Color coding**:
  - 🔴 Red = Dentin
  - 🟢 Green = Enamel
  - 🔵 Blue = Pulp
- **Label** with class name and confidence percentage
- **Title** showing detection summary

## 🎯 Object Detection

### Detection Features

The detection system uses:
- **Sliding window** approach with overlapping patches
- **Confidence filtering** to remove low-quality detections
- **Real-time visualization** with matplotlib

### Usage

```bash
# Interactive mode
python detect_simple.py

# Enter image path when prompted
Enter image path: images/dental_xray.jpg
```

### Detection Parameters

You can modify these in the script:

```python
patch_size = 224          # Size of sliding window
stride = 112              # Overlap between windows
confidence_threshold = 0.6  # Minimum confidence to display
```

### Detection Process

1. **Load Image** - Converts to RGB
2. **Sliding Window** - Extracts 224×224 patches with 112px stride
3. **Classification** - Each patch classified by the model
4. **Filtering** - Only keeps detections above confidence threshold
5. **Selection** - Selects highest confidence detection
6. **Visualization** - Draws bounding box and displays

## 🏗️ Model Architecture

### DenseNet121 Overview

```
Input (224×224×3)
    ↓
DenseNet121 Backbone (Pretrained on ImageNet)
    ↓
Global Average Pooling
    ↓
Fully Connected Layer (1024 → 3)
    ↓
Softmax
    ↓
Output (3 classes)
```

### Model Statistics

- **Total Parameters**: 6,956,931
- **Trainable Parameters**: 6,956,931
- **Model Size**: ~30 MB
- **Input Size**: 224×224×3
- **Output Classes**: 3 (dentin, enamel, pulp)

### Transfer Learning Strategy

1. **Backbone**: DenseNet121 pretrained on ImageNet
2. **Fine-tuning**: All layers trainable with different learning rates
3. **Classifier**: New fully connected layer for 3 classes
4. **Optimization**: AdamW with weight decay

## 📊 Results

### Classification Performance

Based on the trained model:

| Metric | Value |
|--------|-------|
| Validation Accuracy | 98.15% |
| Test Accuracy | ~99% |
| Training Time (30 epochs) | ~5-10 minutes on Mac M4 |

### Sample Predictions

**Example 1: Dentin Detection**
- Image: `1 (2).png`
- Predicted: DENTIN
- Confidence: 99.56%
- Result: ✅ Correct

### Confusion Matrix

The confusion matrix shows excellent performance across all three classes with minimal misclassifications.

### Training Curves

The training history shows:
- Steady decrease in training loss
- Validation accuracy plateaus around 98%
- No significant overfitting
- Early stopping triggered appropriately

## 📂 Project Structure

```
densNet/
├── train.py                    # Main training script
├── detect_simple.py            # Simple detection script
├── requirements.txt            # Python dependencies
├── run_training.sh            # Automated setup and training
├── README.md                  # This file
├── best_densenet_model.pth    # Trained model (generated)
├── training_history.png       # Training curves (generated)
├── confusion_matrix.png       # Test results (generated)
└── segmented_dental_adiography/  # Dataset directory
    ├── train/
    ├── valid/
    └── test/
```

## 🔧 Troubleshooting

### Common Issues

**1. MPS Not Available**
```
Using device: cpu
```
**Solution**: Ensure you're running on Apple Silicon Mac with macOS 12.3+

**2. Out of Memory**
```
RuntimeError: MPS backend out of memory
```
**Solution**: Reduce batch size in `train.py`:
```python
batch_size = 8  # Reduce from 16
```

**3. Model Not Found**
```
FileNotFoundError: Model file not found: best_densenet_model.pth
```
**Solution**: Train the model first:
```bash
python train.py
```

**4. Import Errors**
```
ModuleNotFoundError: No module named 'torch'
```
**Solution**: Activate virtual environment and install dependencies:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**5. Image Not Found**
```
❌ Image not found: path/to/image.png
```
**Solution**: Check the image path and ensure the file exists

### Performance Tips

**For Faster Training:**
- Use MPS acceleration (automatic on Mac M4)
- Increase batch size if you have more RAM
- Reduce number of epochs if needed

**For Better Accuracy:**
- Train for more epochs
- Adjust learning rates
- Increase data augmentation
- Use larger image sizes (requires more memory)

**For Faster Inference:**
- Increase stride (less overlap, faster but less accurate)
- Increase confidence threshold (fewer detections)
- Reduce patch size (not recommended)

## 🎨 Customization

### Modify Training Parameters

Edit `train.py`:

```python
# Change batch size
batch_size = 32  # Default: 16

# Change number of epochs
num_epochs = 50  # Default: 30

# Change learning rates
optimizer = optim.AdamW([
    {'params': [...], 'lr': 5e-5},  # Backbone
    {'params': [...], 'lr': 5e-4}   # Classifier
], weight_decay=1e-4)

# Change early stopping patience
patience = 10  # Default: 7
```

### Modify Detection Parameters

Edit `detect_simple.py`:

```python
# Change confidence threshold
confidence_threshold = 0.7  # Default: 0.6

# Change sliding window parameters
patch_size = 256  # Default: 224
stride = 128      # Default: 112
```

### Add New Classes

To add more dental structure classes:

1. Update `CLASS_NAMES` in both scripts
2. Add corresponding folders to dataset
3. Update model output layer:
```python
model.classifier = nn.Linear(num_features, 4)  # For 4 classes
```

## 📈 Advanced Usage

### Batch Prediction

Create a script to process multiple images:

```python
import os
from detect_simple import detect

image_dir = "path/to/images"
for img_file in os.listdir(image_dir):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_dir, img_file)
        result, detection = detect(img_path)
        print(f"{img_file}: {detection[4]} ({detection[5]*100:.1f}%)")
```

### Export Model

Export the model for deployment:

```python
import torch

# Load model
model = load_model('best_densenet_model.pth')

# Export to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')
```

### Model Evaluation

Evaluate on custom test set:

```python
from train import validate, DentalRadiographyDataset

# Load test dataset
test_dataset = DentalRadiographyDataset('custom_test_dir', transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=16)

# Evaluate
test_loss, test_acc, preds, labels = validate(model, test_loader, criterion, device)
print(f"Test Accuracy: {test_acc:.2f}%")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd densNet

# Create development environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Make changes and test
python train.py
python detect_simple.py
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- DenseNet architecture from [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- PyTorch framework and pretrained models
- Apple Silicon MPS acceleration
- Kaggle dental radiography dataset

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

## 🌍 Documentation

- [English Documentation](README.md) - Full documentation in English
- [مستندات فارسی](README_FA.md) - مستندات کامل به زبان فارسی

---

**Built with ❤️ for dental healthcare professionals and researchers**

*Optimized for Apple Silicon (Mac M4) with MPS acceleration*
