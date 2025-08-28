import numpy as np
import argparse
from DeepNeuralNetwork_Numpy import DeepNeuralNetwork_Numpy
from DeepNeuralNetwork_Pytorch import DeepNeuralNetwork_Pytorch


NODES = [2, 8, 8, 1]
MODEL = "pytorch"        ### pytorch or numpy
OPT = "sgd"             ### "sgd", "momentum", "adam"
ACT = "tanh"             ### "sigmoid", "relu", "tanh"
L_R = 0.01
EPOCH = 10


# Define XOR dataset
def generate_xor_data(n_samples=1000):
    X = np.random.randint(0, 2, size=(n_samples, 2))
    Y = np.logical_xor(X[:, 0], X[:, 1]).astype(int).reshape(-1, 1)
    return X.astype(np.float32), Y

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

    print("Generating XOR data...")
    x, y = generate_xor_data(1000)

    # Train-test split
    split = int(0.8 * len(x))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    print("Training data:", x_train.shape, y_train.shape)
    print("Test data:", x_test.shape, y_test.shape)

    print("Start training...")
    # Model selection
    if args.model.lower() == "pytorch":
        model = DeepNeuralNetwork_Pytorch(nodes=args.nodes, activation=args.activation)
    else:
        model = DeepNeuralNetwork_Numpy(nodes=args.nodes, activation=args.activation)
    model.Initialization()
    model.Train(
        x_train, y_train, x_test, y_test,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        learning_rate=args.l_rate,
        beta=args.beta,
        epochs=args.epochs
    )

    # Evaluate
    preds = model.Predict(x_test)
    acc = np.mean(preds == y_test.flatten())
    print(f"Final test accuracy: {acc:.4f}")

    # Show predictions on test set
    for i in range(10):
        print(f"Input: {x_test[i]} | Predicted: {preds[i]} | Actual: {y_test[i][0]}")
