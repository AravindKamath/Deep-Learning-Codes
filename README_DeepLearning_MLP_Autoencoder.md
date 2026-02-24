
# Deep Learning Implementation from Scratch
## MLP + Autoencoder + Outlier Detection (NumPy-Based)

---

## 1. Project Overview

This project implements fundamental deep learning models **from scratch using NumPy**, without relying on high-level deep learning frameworks such as PyTorch or Keras for model construction.

The combined script includes:

- Multi-Layer Perceptron (MLP) for MNIST classification
- Autoencoder (dense) for image reconstruction
- Outlier detection using reconstruction error
- Training loss and accuracy visualization
- Reconstruction error histogram generation

The objective of this implementation is to understand core deep learning concepts such as:
- Forward propagation
- Backpropagation
- Gradient-based optimization
- Representation learning
- Reconstruction-based anomaly detection

---

## 2. File Included

- `deeplearning_mlp_autoencoder_combined.py`  
  → Complete end-to-end script containing all components.

---

## 3. Requirements

### Python Version
Python 3.8 or higher is recommended.

### Required Libraries

Install the required dependencies using:

```bash
pip install numpy matplotlib tensorflow
```

Required packages:

- numpy
- matplotlib
- tensorflow (only used to load MNIST dataset)

---

## 4. How to Run the Script

### Step 1: Navigate to the project directory

```bash
cd path/to/project/folder
```

### Step 2: Run the script

```bash
python deeplearning_mlp_autoencoder_combined.py
```

---

## 5. What Happens During Execution

### MLP Training
- Loads MNIST dataset
- Flattens images into 784-dimensional vectors
- Trains a 2-layer MLP (784 → 128 → 10)
- Prints:
  - Training loss per epoch
  - Training accuracy per epoch
  - Final test accuracy

### Generated Output Files
After execution, the following plots will be saved:

- `mlp_loss.png`  
- `mlp_accuracy.png`  
- `autoencoder_loss.png`  
- `reconstruction_error_hist.png`  

These files are automatically saved in the same directory.

---

## 6. Model Architecture Details

### Multi-Layer Perceptron
- Input Layer: 784 neurons
- Hidden Layer: 128 neurons (ReLU activation)
- Output Layer: 10 neurons (Softmax activation)
- Loss Function: Cross-Entropy

### Autoencoder
- Architecture: 784 → 128 → 32 → 128 → 784
- Hidden Activations: ReLU
- Output Activation: Sigmoid
- Loss Function: Mean Squared Error (MSE)

### Outlier Detection
- Reconstruction error computed per sample
- 99th percentile used as anomaly threshold
- Histogram plot visualizes distribution

---

## 7. Expected Results

- MLP Training Accuracy: ~95–97%
- MLP Test Accuracy: ~94–96%
- Autoencoder Reconstruction Loss: ~0.02–0.03
- Stable convergence without numerical instability

---

## 8. Notes

- The script uses CPU execution only.
- No GPU acceleration is required.
- All gradients are implemented manually (no automatic differentiation).
- The code is written for educational clarity rather than production optimization.

---

## 9. Troubleshooting

### If TensorFlow is not installed:
```bash
pip install tensorflow
```

### If matplotlib plots do not save:
Ensure you have write permission in the current directory.

---

## 10. Academic Use

This implementation is intended for educational purposes and academic submission.  
All core learning mechanisms are implemented manually to reinforce conceptual understanding of deep learning fundamentals.

---

End of README
