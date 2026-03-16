
"""
Deep Learning Combined Implementation
MLP + Autoencoder + Outlier Detection
Author: Combined Script Version
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# -------------------------
# Utility Functions
# -------------------------

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

def mse(x, x_hat):
    return np.mean((x - x_hat) ** 2)

# -------------------------
# Load and Preprocess MNIST
# -------------------------

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

# -------------------------
# MLP Implementation
# -------------------------

class MLP:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, y_true, lr=0.01):
        m = X.shape[0]
        dZ2 = self.A2 - y_true
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dZ1 = dZ2 @ self.W2.T * relu_derivative(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# -------------------------
# Train MLP
# -------------------------

mlp = MLP()
epochs = 5
batch_size = 64
learning_rate = 0.01

losses = []
accuracies = []

for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train_oh[i:i+batch_size]
        mlp.forward(X_batch)
        mlp.backward(X_batch, y_batch, learning_rate)

    preds = mlp.predict(X_train)
    acc = np.mean(preds == y_train)
    loss = cross_entropy(y_train_oh, mlp.forward(X_train))

    losses.append(loss)
    accuracies.append(acc)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Train Acc: {acc:.4f}")

test_preds = mlp.predict(X_test)
test_acc = np.mean(test_preds == y_test)
print("Test Accuracy:", test_acc)

# Plot Loss
plt.figure()
plt.plot(losses)
plt.title("MLP Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("mlp_loss.png")

# Plot Accuracy
plt.figure()
plt.plot(accuracies)
plt.title("MLP Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("mlp_accuracy.png")

# -------------------------
# Autoencoder Implementation
# -------------------------

class Autoencoder:
    def __init__(self, input_size=784, hidden_size=128, latent_size=32):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, latent_size) * 0.01
        self.b2 = np.zeros((1, latent_size))
        self.W3 = np.random.randn(latent_size, hidden_size) * 0.01
        self.b3 = np.zeros((1, hidden_size))
        self.W4 = np.random.randn(hidden_size, input_size) * 0.01
        self.b4 = np.zeros((1, input_size))

    def forward(self, X):
        self.A1 = relu(X @ self.W1 + self.b1)
        self.Z = relu(self.A1 @ self.W2 + self.b2)
        self.A3 = relu(self.Z @ self.W3 + self.b3)
        self.X_hat = 1 / (1 + np.exp(-(self.A3 @ self.W4 + self.b4)))
        return self.X_hat

    def train(self, X, epochs=5, lr=0.01):
        losses = []
        for epoch in range(epochs):
            X_hat = self.forward(X)
            loss = mse(X, X_hat)
            losses.append(loss)
            print(f"Autoencoder Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
        return losses

# Train Autoencoder
ae = Autoencoder()
ae_losses = ae.train(X_train[:5000], epochs=5)

plt.figure()
plt.plot(ae_losses)
plt.title("Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.savefig("autoencoder_loss.png")

# -------------------------
# Outlier Detection
# -------------------------

X_hat = ae.forward(X_test[:1000])
reconstruction_errors = np.mean((X_test[:1000] - X_hat) ** 2, axis=1)
threshold = np.percentile(reconstruction_errors, 99)

plt.figure()
plt.hist(reconstruction_errors, bins=50)
plt.axvline(threshold)
plt.title("Reconstruction Error Distribution")
plt.savefig("reconstruction_error_hist.png")

print("Outlier Detection Threshold (99th percentile):", threshold)
print("Script execution completed successfully.")
