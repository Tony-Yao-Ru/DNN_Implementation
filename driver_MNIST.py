import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from DeepNeuralNetwork_Numpy import DeepNeuralNetwork_Numpy
from DeepNeuralNetwork_Pytorch import DeepNeuralNetwork_Pytorch

def one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]

NODES = [784, 64, 64, 10]
MODEL = "pytorch"        ### pytorch or numpy
OPT = "momentum"             ### "sgd", "momentum", "adam"
ACT = "sigmoid"             ### "sigmoid", "relu", "tanh"
L_R = 0.001
EPOCH = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deep neural network on MNIST using NumPy or PyTorch.")
    parser.add_argument("--nodes", nargs="+", type=int, default=NODES,
                        help="List of layer sizes including input and output. Example: --nodes 784 256 128 10")
    parser.add_argument("--model", default=MODEL, choices=["numpy", "pytorch"],
                        help="Choose the backend for training: 'numpy' (pure NumPy) or 'pytorch' (uses PyTorch).")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini-batch size used during training.")
    parser.add_argument("--optimizer", default=OPT, choices=["sgd", "momentum", "adam"],
                        help="Optimization algorithm to use: 'sgd', 'momentum', or 'adam'.")
    parser.add_argument("--activation", default=ACT, choices=["sigmoid", "relu", "tanh"],
                        help="Activation function to use in hidden layers: 'sigmoid', 'relu', or 'tanh'.")
    parser.add_argument("--l_rate", type=float, default=L_R,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--beta", type=float, default=0.9,
                        help="Beta value used for momentum or Adam optimizer.")
    parser.add_argument("--epochs", type=int, default=EPOCH,
                        help="Number of training epochs.")

    args = parser.parse_args()

    print("Loading data...")
    mnist_data = fetch_openml("mnist_784", as_frame=False)
    x = mnist_data["data"].astype("float32") / 255.0
    y = mnist_data["target"].astype("int32")

    num_classes = 10
    y_encoded = one_hot(y, num_classes)

    x_train, x_test = x[:60000], x[60000:]
    y_train, y_test = y_encoded[:60000], y_encoded[60000:]

    shuffle_idx = np.random.permutation(60000)
    x_train, y_train = x_train[shuffle_idx], y_train[shuffle_idx]

    # Model selection
    if args.model.lower() == "pytorch":
        model = DeepNeuralNetwork_Pytorch(nodes=args.nodes, activation=args.activation)
        y_train_model = np.argmax(y_train, axis=1)  # integer labels for PyTorch
        y_test_model = np.argmax(y_test, axis=1)
    else:
        model = DeepNeuralNetwork_Numpy(nodes=args.nodes, activation=args.activation)
        y_train_model = y_train  # one-hot labels for NumPy
        y_test_model = y_test


    model.Initialization()
    model.Train(
        x_train, y_train_model, x_test, y_test_model,
        batch_size=int(args.batch_size),
        optimizer=args.optimizer,
        learning_rate=float(args.l_rate),
        beta=float(args.beta),
        epochs=int(args.epochs)
    )

    preds = model.Predict(x_test)
    true_labels = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
    acc = np.mean(preds == true_labels)
    print(f"Final test accuracy: {acc:.4f}")

    for i in range(5):
        image = x_test[i].reshape(28, 28)
        plt.imshow(image, cmap="gray")
        plt.title(f"Predicted: {preds[i]} | Actual: {true_labels[i]}")
        plt.axis("off")
        plt.show()
