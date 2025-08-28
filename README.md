# 🧠 Deep Neural Network from Scratch in NumPy

This repository contains a custom implementation of a deep neural network (DNN) built entirely from scratch using NumPy. It’s designed to be a lightweight and educational framework for understanding the internals of neural networks without relying on external machine learning libraries.

---

## 🚀 Features

- Support for arbitrary number of layers and nodes  
- Configurable activation functions:  
  - **Sigmoid**  
  - **ReLU**  
  - **Tanh**  
- Output layers for:  
  - **Binary classification** (sigmoid)  
  - **Multi-class classification** (softmax)  
- Optimization algorithms:  
  - **Stochastic Gradient Descent (SGD)**  
  - **Momentum**  
  - **Adam**  
- Implements forward and backward propagation  
- Mini-batch training  
- Evaluation metrics:  
  - Accuracy  
  - Cross-entropy loss  

---

## 🧩 Code Overview

The main class is:

```python
DeepNeuralNetwork(nodes, activation='sigmoid')

