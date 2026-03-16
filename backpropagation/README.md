# Backpropagation in ANN, CNN, and RNN

Mini Seminar – Deep Learning

**Team Members**

* Member 1: <Your Name>
* Member 2: <Teammate Name>

---

# Table of Contents

1. Introduction
2. Backpropagation Overview
3. Backpropagation in Artificial Neural Networks (ANN)
4. Backpropagation in Convolutional Neural Networks (CNN)
5. Backpropagation in Recurrent Neural Networks (RNN)
6. Differences Between ANN, CNN, and RNN Backpropagation
7. Code Implementation
8. Output of Code
9. References
10. Conclusion

---

# 1. Introduction

Deep learning models learn by adjusting their internal parameters (weights) based on the error between predicted and actual outputs.
The algorithm responsible for this learning process is **Backpropagation**.

Backpropagation computes gradients of the loss function with respect to the network parameters and updates them using **gradient descent**.

The training process consists of two main phases:

1. **Forward Propagation** – compute predictions
2. **Backward Propagation** – compute gradients and update weights

Although the fundamental concept remains the same, the **implementation of backpropagation differs across ANN, CNN, and RNN** due to differences in network architecture.

---

# 2. Backpropagation Overview

Backpropagation relies on the **Chain Rule of Calculus** to compute gradients efficiently.

### General Gradient Formula

$$
\frac{\partial L}{\partial w}
=

\frac{\partial L}{\partial y}
\cdot
\frac{\partial y}{\partial z}
\cdot
\frac{\partial z}{\partial w}
$$

Where

* $L$ = Loss function
* $w$ = network weight
* $y$ = output
* $z$ = weighted input

---

### Weight Update Rule

Weights are updated using gradient descent:

$$
w_{new} = w_{old} - \eta \frac{\partial L}{\partial w}
$$

Where

* $\eta$ = learning rate

---

# 3. Backpropagation in Artificial Neural Networks (ANN)

## Architecture

Artificial Neural Networks consist of **fully connected layers**, meaning every neuron in one layer connects to all neurons in the next layer.

---

## Forward Propagation

The neuron performs a linear transformation:

$$
z = Wx + b
$$

Where

* $W$ = weight
* $x$ = input
* $b$ = bias

Activation using the sigmoid function:

$$
a = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

---

## Loss Function

A common loss function is **Mean Squared Error (MSE)**:

$$
L = \frac{1}{2}(y - \hat{y})^2
$$

Where

* $y$ = actual output
* $\hat{y}$ = predicted output

---

## Gradient Computation

Using the chain rule:

$$
\frac{\partial L}{\partial W}
=

\frac{\partial L}{\partial \hat{y}}
\cdot
\frac{\partial \hat{y}}{\partial z}
\cdot
\frac{\partial z}{\partial W}
$$

---

## Algorithm for ANN Backpropagation

1. Initialize weights randomly
2. Perform forward propagation
3. Compute loss
4. Calculate gradients using chain rule
5. Update weights using gradient descent
6. Repeat until loss decreases

---

## Calculation Example

Input

$$
x = 1
$$

Weight

$$
W = 0.5
$$

Bias

$$
b = 0
$$

Forward pass:

$$
z = (0.5)(1) + 0 = 0.5
$$

Sigmoid output:

$$
\hat{y} = \frac{1}{1 + e^{-0.5}} \approx 0.62
$$

Loss:

$$
L = \frac{1}{2}(1 - 0.62)^2 \approx 0.072
$$

Backpropagation updates the weight to reduce the error.

---

# 4. Backpropagation in Convolutional Neural Networks (CNN)

## Architecture

CNNs are designed for **spatial data such as images**.

Instead of fully connected neurons, CNNs use:

* Convolution layers
* Filters (kernels)
* Feature maps
* Pooling layers

---

## Convolution Operation

Feature maps are generated using convolution:

$$
F(i,j) =
\sum_m \sum_n
I(i+m,j+n)K(m,n)
$$

Where

* $I$ = input image
* $K$ = kernel
* $F$ = feature map

---

## Gradient Computation

During backpropagation, gradients are computed for the filter weights:

$$
\frac{\partial L}{\partial K} = I * \delta
$$

Where

* $I$ = input image
* $\delta$ = error gradient

---

## Kernel Update Rule

$$
K_{new} = K_{old} - \eta \frac{\partial L}{\partial K}
$$

---

## Algorithm for CNN Backpropagation

1. Apply convolution filters to input image
2. Generate feature maps
3. Apply activation functions
4. Compute loss
5. Compute gradient for filters
6. Update filter weights

---

## Calculation Example

Input Image

```
1 2 3
4 5 6
7 8 9
```

Kernel

```
1 0
0 -1
```

Example convolution calculation:

$$
(1×1 + 2×0 + 4×0 + 5×(-1)) = -4
$$

Feature map:

```
-4 -4
-4 -4
```

Backpropagation updates the kernel values accordingly.

---

# 5. Backpropagation in Recurrent Neural Networks (RNN)

## Architecture

RNNs are designed for **sequential data**, such as:

* text
* speech
* time series

They maintain a **hidden state** that carries information across time steps.

---

## Hidden State Equation

At time step $t$:

$$
h_t = \tanh(Wx_t + Uh_{t-1} + b)
$$

Where

* $x_t$ = input at time $t$
* $h_{t-1}$ = previous hidden state
* $W$ = input weight matrix
* $U$ = recurrent weight matrix

---

## Output Equation

$$
y_t = Vh_t
$$

---

## Loss Function

Total loss across sequence:

$$
L = \sum_{t=1}^{T}(y_t - \hat{y}_t)^2
$$

---

## Backpropagation Through Time (BPTT)

RNN training uses **Backpropagation Through Time**, where gradients propagate backward across time steps.

$$
\frac{\partial L}{\partial W}
=

\sum_{t=1}^{T}
\frac{\partial L_t}{\partial W}
$$

---

## Algorithm for RNN Backpropagation

1. Process input sequence through time
2. Compute hidden states
3. Compute loss at each time step
4. Unroll network across time
5. Backpropagate gradients through time
6. Update weights

---

## Calculation Example

Input sequence:

$$
x_1 = 1, \quad x_2 = 2, \quad x_3 = 3
$$

Hidden state update:

$$
h_t = \tanh(Wx_t + Uh_{t-1})
$$

Each step contributes to the total loss, and gradients accumulate across time.

---

# 6. Differences Between ANN, CNN, and RNN Backpropagation

| Feature           | ANN                      | CNN                                 | RNN                          |
| ----------------- | ------------------------ | ----------------------------------- | ---------------------------- |
| Network Structure | Fully connected          | Convolution filters                 | Recurrent connections        |
| Data Type         | Tabular data             | Image data                          | Sequential data              |
| Gradient Flow     | Layer-to-layer           | Through convolution filters         | Through time                 |
| Parameter Sharing | No                       | Yes                                 | Yes                          |
| Training Method   | Standard Backpropagation | Backpropagation through convolution | Backpropagation Through Time |

---

# 7. Code Implementation

The implementations are provided in the repository:

```
code/ann_backpropagation.py
code/cnn_backpropagation.py
code/rnn_backpropagation.py
```

Each file demonstrates the training process using backpropagation.

---

# 8. Expected Output of Code

### ANN

```
Updated weight: [[0.50445]]
```

---

### CNN

Feature Map

```
[[-4 -4]
 [-4 -4]]
```

Updated Kernel

```
[[ 0.999 -0.001]
 [-0.001 -1.001]]
```

---

### RNN

Example Output (values vary due to random initialization)

```
Hidden states: [0.72, 0.88, 0.96]

Updated Wx: 0.52
Updated Wh: -0.31
```

---

# 9. References

GeeksforGeeks – Backpropagation in Neural Networks
https://www.geeksforgeeks.org/backpropagation-in-neural-network/

GeeksforGeeks – Convolutional Neural Networks
https://www.geeksforgeeks.org/convolutional-neural-network-cnn/

GeeksforGeeks – Recurrent Neural Networks
https://www.geeksforgeeks.org/recurrent-neural-network-rnn/

---

# 10. Conclusion

Backpropagation is the core mechanism that allows neural networks to learn from data.

Although the underlying concept remains the same, its implementation varies across different architectures:

* ANN uses **standard gradient propagation through layers**
* CNN adapts backpropagation to **learn convolution filters**
* RNN extends backpropagation across **time using Backpropagation Through Time**

Understanding these differences helps in selecting the appropriate architecture for different machine learning problems.
