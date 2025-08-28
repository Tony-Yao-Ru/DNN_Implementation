# DNN_Implementation

This repository offers a custom implementation of a Deep Neural Network (DNN) built entirely from scratch using NumPy.  
It serves as an educational tool to understand the inner workings of neural networks without relying on external machine learning libraries.

---

## 🚀 Features

- **Customizable Architecture**: Supports an arbitrary number of layers and nodes, allowing for flexible network designs.
- **Activation Functions**:
  - Sigmoid
  - ReLU
  - Tanh
- **Output Layers**:
  - Binary classification using sigmoid activation
  - Multi-class classification using softmax activation
- **Optimization Algorithms**:
  - Stochastic Gradient Descent (SGD)
  - Momentum-based SGD
  - Adaptive methods (e.g., Adam)
- **Loss Functions**:
  - Mean Squared Error (MSE) for regression tasks
  - Cross-Entropy Loss for classification tasks
- **Regularization**:
  - L2 regularization to prevent overfitting

---

## 🧪 Modules

### 1. `DeepNeuralNetwork.py`
The core module containing the implementation of the DNN class.  
This class encompasses methods for:
- Initializing the network architecture
- Forward propagation
- Backward propagation
- Training the model
- Evaluating performance

### 2. `driver_MNIST.py`
A script designed to train and evaluate the DNN on the MNIST dataset, which consists of handwritten digits.  
It demonstrates the network's capability to perform image classification tasks.

### 3. `driver_XOR.py`
A script that showcases the DNN's ability to solve the XOR problem, a classic example in neural network training.  
It highlights the network's capacity to learn non-linear decision boundaries.

---

## 📦 Requirements

To run this implementation, ensure you have the following Python packages installed:
- NumPy
- Matplotlib (for visualization)

Install via pip:

```bash
pip install numpy matplotlib
```

---

## ⚙️ Usage

1. Clone the repository:

```bash
git clone https://github.com/Tony-Yao-Ru/DNN_Implementation.git
cd DNN_Implementation
```

2. Run the desired driver script:

- For MNIST classification:

```bash
python driver_MNIST.py=
```

- For XOR problem:

```bash
python driver_XOR.py
```
