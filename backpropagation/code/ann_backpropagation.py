import numpy as np

# -------------------------------
# Sigmoid activation function
# -------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# -------------------------------
# Input and expected output
# -------------------------------
X = np.array([[1]])     # input
y = np.array([[1]])     # target output

# -------------------------------
# Initialize weight and bias
# -------------------------------
W = np.array([[0.5]])
b = 0

learning_rate = 0.1

print("Initial Weight:", W)

# -------------------------------
# Forward Propagation
# -------------------------------
z = np.dot(X, W) + b
y_pred = sigmoid(z)

print("Predicted Output:", y_pred)

# -------------------------------
# Loss Calculation (MSE)
# -------------------------------
loss = 0.5 * (y - y_pred) ** 2
print("Loss:", loss)

# -------------------------------
# Backpropagation
# -------------------------------
dL_dy = -(y - y_pred)
dy_dz = sigmoid_derivative(y_pred)

gradient = np.dot(X.T, dL_dy * dy_dz)

# -------------------------------
# Weight Update
# -------------------------------
W = W - learning_rate * gradient

print("Updated Weight:", W)