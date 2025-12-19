"""
Simple Object Detection - Easy to use version
Shows only the highest confidence detection and displays the image
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw
import numpy as np
import os
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = 'best_densenet_model.pth'
CLASS_NAMES = ['dentin', 'enamel', 'pulp']
CLASS_COLORS = {
    'dentin': (255, 0, 0),    # Red
    'enamel': (0, 255, 0),    # Green
    'pulp': (0, 0, 255)       # Blue
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
def load_model():
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 3)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model

print("Loading model...")
model = load_model()
print("✅ Model loaded!")

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect(image_path, confidence_threshold=0.6):
    """Simple detection function - returns only highest confidence detection"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    print(f"Image size: {w}x{h}")
    
    # Sliding window
    patch_size = 224
    stride = 112
    detections = []
    
    print("Scanning image...")
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            
            # Predict
            input_tensor = transform(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, pred = torch.max(probs, 0)
            
            if conf.item() >= confidence_threshold:
                class_name = CLASS_NAMES[pred.item()]
                detections.append((x, y, x + patch_size, y + patch_size, class_name, conf.item()))
    
    # Find highest confidence detection
    if not detections:
        print("❌ No detections found above confidence threshold")
        return image, None
    
    # Sort by confidence and get the highest
    detections.sort(key=lambda x: x[5], reverse=True)
    best_detection = detections[0]
    x1, y1, x2, y2, class_name, conf = best_detection
    
    print(f"\n✅ Highest confidence detection:")
    print(f"   Class: {class_name.upper()}")
    print(f"   Confidence: {conf*100:.2f}%")
    print(f"   Location: ({x1}, {y1}) to ({x2}, {y2})")
    
    # Draw only the highest confidence detection
    result = image.copy()
    draw = ImageDraw.Draw(result)
    
    color = CLASS_COLORS[class_name]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    # Draw label with background
    label_text = f"{class_name.upper()} {conf*100:.1f}%"
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
    
    # Get text size for background
    bbox = draw.textbbox((0, 0), label_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Draw label background
    label_y = max(0, y1 - text_height - 4)
    draw.rectangle(
        [x1, label_y, x1 + text_width + 8, y1],
        fill=color
    )
    
    # Draw label text
    draw.text(
        (x1 + 4, label_y),
        label_text,
        fill=(255, 255, 255),
        font=font
    )
    
    return result, best_detection

# Example usage
if __name__ == '__main__':
    image_path = input("Enter image path: ").strip()
    
    if os.path.exists(image_path):
        result, detection = detect(image_path)
        
        if detection is not None:
            # Display the image
            plt.figure(figsize=(12, 8))
            plt.imshow(result)
            plt.axis('off')
            plt.title(f"Detection: {detection[4].upper()} ({detection[5]*100:.1f}% confidence)", 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            print("\n✅ Image displayed!")
        else:
            # Show original image if no detection
            plt.figure(figsize=(12, 8))
            plt.imshow(result)
            plt.axis('off')
            plt.title("No detections found", fontsize=14)
            plt.tight_layout()
            plt.show()
    else:
        print(f"❌ Image not found: {image_path}")

