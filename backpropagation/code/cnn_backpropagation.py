import numpy as np

# -------------------------------
# Input Image
# -------------------------------
image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("Input Image:")
print(image)

# -------------------------------
# Convolution Kernel
# -------------------------------
kernel = np.array([
    [1, 0],
    [0, -1]
])

print("\nInitial Kernel:")
print(kernel)

# -------------------------------
# Convolution Operation
# -------------------------------
output = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        region = image[i:i+2, j:j+2]
        output[i, j] = np.sum(region * kernel)

print("\nFeature Map:")
print(output)

# -------------------------------
# Simulated Gradient (for demo)
# -------------------------------
gradient = np.ones_like(kernel) * 0.01

# -------------------------------
# Kernel Update
# -------------------------------
learning_rate = 0.1
kernel = kernel - learning_rate * gradient

print("\nUpdated Kernel:")
print(kernel)