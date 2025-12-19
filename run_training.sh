#!/bin/bash

# Dental Radiography Classification Training Script
# Optimized for Mac M4

echo "=========================================="
echo "Dental Radiography Classification"
echo "Training on Mac M4"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade requirements
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Run training
echo ""
echo "Starting training..."
echo "=========================================="
python train.py

echo ""
echo "Training completed!"
echo "Check the following files:"
echo "  - best_densenet_model.pth (saved model)"
echo "  - training_history.png (training curves)"
echo "  - confusion_matrix.png (test set results)"

