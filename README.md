# Deep Learning Models: CNN, RNN Variants, and GAN

This project implements and evaluates several deep learning architectures across different machine learning tasks. The objective is to study how different neural network models perform on image classification, sequence learning, and generative modeling problems.

The project was developed as part of a Deep Learning coursework assignment.

---

## Models Implemented

### 1. Convolutional Neural Network (CNN)
A custom CNN architecture was implemented for image classification on the Fashion-MNIST dataset.  
The model includes convolutional layers, ReLU activations, batch normalization, max pooling, and dropout regularization.

### 2. Transfer Learning (ResNet18)
A pretrained **ResNet18** model was used for comparison with the custom CNN.  
The model was adapted for grayscale Fashion-MNIST images and the final classification layer was retrained for the task.

### 3. Recurrent Neural Networks for Text Classification
Three sequence models were implemented using **TensorFlow/Keras**:

- Simple RNN  
- LSTM  
- GRU  

These models were trained on the **IMDB movie review dataset** to perform binary sentiment classification.

### 4. Generative Adversarial Network (GAN)
A GAN was implemented to generate synthetic **Fashion-MNIST images**.

The model consists of:

**Generator**
- Generates images from random noise

**Discriminator**
- Distinguishes between real images and generated images

---

## Project Structure

```
DeepLearning-Models/
│
├── notebooks/
│   └── deep_learning_experiments.ipynb
│
├── figures/
│   ├── cnn_training_validation_metrics.png
│   ├── cnn_confusion_matrix.png
│   ├── rnn_validation_accuracy.png
│   ├── rnn_validation_loss.png
│   ├── gan_loss_curve.png
│   └── gan_final_samples.png
│
├── report/
│   └── deep_learning_assignment.pdf
│
└── README.md
```

---

## Requirements

The experiments were conducted using **Google Colab with GPU support**.

Main dependencies:

- Python 3.x  
- PyTorch  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

Install them locally using:

```
pip install torch torchvision tensorflow matplotlib seaborn scikit-learn
```

---

## How to Run the Project

### Option 1 — Run Using Google Colab (Recommended)

1. Open the notebook located in the `notebooks` folder.

```
notebooks/deep_learning_experiments.ipynb
```

2. Upload the notebook to **Google Colab**

3. Enable GPU:

```
Runtime → Change runtime type → GPU
```

4. Run the cells sequentially from top to bottom.

---

### Option 2 — Run Locally

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/deep-learning-assignment.git
```

Navigate into the project folder:

```
cd deep-learning-assignment
```

Install dependencies:

```
pip install torch torchvision tensorflow matplotlib seaborn scikit-learn
```

Open the notebook:

```
jupyter notebook
```

Run all cells in order.

---

## Results

### CNN Performance

| Model | Test Accuracy |
|------|---------------|
| Custom CNN | **92.05%** |
| ResNet18 | **67.29%** |

### RNN Model Comparison

| Model | Test Accuracy |
|------|---------------|
| Simple RNN | 81.12% |
| LSTM | 85.21% |
| GRU | **86.81%** |

### GAN

The GAN successfully learned the distribution of Fashion-MNIST images and produced synthetic clothing samples.  
Training stability was observed when discriminator accuracy stabilized near **50%**, indicating balanced adversarial learning.

---

## Generated Samples

Example GAN generated images:

![GAN Samples](figures/gan_final_samples.png)

---

## Key Observations

- The **custom CNN** performed significantly better than the pretrained ResNet18 model on Fashion-MNIST.
- **LSTM and GRU** outperformed the vanilla RNN in sentiment classification tasks.
- The **GAN successfully captured the general structure of Fashion-MNIST images**, generating recognizable clothing patterns.

---

## Author

K. Aravind Kamath  
Information Science and Engineering  
NMAM Institute of Technology

---

## License

This project is intended for academic and educational purposes.