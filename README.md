# 🧠 DNN_Implementation

This repository provides a customizable deep neural network (DNN) implementation using both **NumPy (from scratch)** and **PyTorch**.  
It serves as an educational framework for understanding how neural networks work under the hood and offers a modular design to compare manual vs. library-based training.

---

## 🚀 Features

- **Customizable Architecture**: Define arbitrary layers and node sizes using command-line arguments.
- **Activation Functions**:
  - `sigmoid`
  - `relu`
  - `tanh`
- **Output Layers**:
  - Binary classification with sigmoid
  - Multi-class classification with softmax
- **Optimization Algorithms**:
  - Stochastic Gradient Descent (SGD)
  - Momentum-based SGD
  - Adam optimizer
- **Loss Functions**:
  - Binary Cross-Entropy
  - Multi-Class Cross-Entropy
- **Backends**:
  - Pure NumPy (manual forward/backward pass)
  - PyTorch (hardware-accelerated training)
- **Visualization**:
  - MNIST image predictions
  - XOR problem predictions printed in console

---

## 📁 Modules

### 1. `DeepNeuralNetwork_Numpy.py`
Implements a full feedforward neural network from scratch using NumPy, including:
- Layer initialization
- Forward & backward propagation
- Optimizers (`sgd`, `momentum`, `adam`)
- Training & prediction routines

### 2. `DeepNeuralNetwork_Pytorch.py`
A PyTorch-based implementation of the same DNN model, using `torch.nn.Module`.  
Supports GPU training and uses native PyTorch utilities like `DataLoader`, `CrossEntropyLoss`, etc.

### 3. `driver_MNIST.py`
Script to train the DNN on the **MNIST** dataset (handwritten digits).  
Supports both NumPy and PyTorch backends.

### 4. `driver_XOR.py`
Script to train the DNN on the classic **XOR problem**.  
Verifies the ability to learn non-linear relationships.

---

## 🧪 Requirements

Install required Python packages:

```bash
pip install numpy matplotlib scikit-learn torch
```

---

## ⚙️ Usage

1. Clone the repository

```bash
git clone https://github.com/Tony-Yao-Ru/DNN_Implementation.git
cd DNN_Implementation
```

2. Train on MNIST

```bash
python driver_MNIST.py \
    --nodes 784 128 64 10 \
    --model numpy \
    --activation relu \
    --optimizer adam \
    --l_rate 0.001 \
    --epochs 20 \
    --batch_size 64
```

3. Train on XOR

```bash
python driver_XOR.py \
    --nodes 2 8 8 1 \
    --model pytorch \
    --activation tanh \
    --optimizer momentum \
    --l_rate 0.01 \
    --epochs 30 \
    --batch_size 32
```

---

## 🔧 Customization

You can modify architecture and behavior using command-line flags:

| Argument       | Description                            | Example                   |
|----------------|----------------------------------------|---------------------------|
| `--nodes`      | Layer sizes incl. input/output         | `--nodes 784 128 64 10`   |
| `--model`      | Backend: `numpy` or `pytorch`          | `--model pytorch`         |
| `--activation` | Activation: `sigmoid`, `relu`, `tanh`  | `--activation tanh`       |
| `--optimizer`  | Optimizer: `sgd`, `momentum`, `adam`   | `--optimizer adam`        |
| `--l_rate`     | Learning rate                          | `--l_rate 0.001`          |
| `--epochs`     | Number of training epochs              | `--epochs 20`             |
| `--batch_size` | Size of mini-batches                   | `--batch_size 64`         |
