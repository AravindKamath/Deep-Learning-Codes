import numpy as np

# -------------------------------
# Initialize weights
# -------------------------------
Wx = np.random.randn()
Wh = np.random.randn()

print("Initial Wx:", Wx)
print("Initial Wh:", Wh)

# -------------------------------
# Initial hidden state
# -------------------------------
h_prev = 0

# -------------------------------
# Input sequence
# -------------------------------
inputs = [1, 2, 3]

hidden_states = []

# -------------------------------
# Forward Pass
# -------------------------------
for x in inputs:
    h = np.tanh(Wx * x + Wh * h_prev)
    hidden_states.append(h)
    h_prev = h

print("\nHidden States:")
print(hidden_states)

# -------------------------------
# Simulated Gradient
# -------------------------------
gradient = 0.01
learning_rate = 0.1

# -------------------------------
# Weight Update
# -------------------------------
Wx = Wx - learning_rate * gradient
Wh = Wh - learning_rate * gradient

print("\nUpdated Wx:", Wx)
print("Updated Wh:", Wh)