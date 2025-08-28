import numpy as np
from DeepNeuralNetwork import DeepNeuralNetwork
import argparse

# Define XOR dataset
def generate_xor_data(n_samples=1000):
    X = np.random.randint(0, 2, size=(n_samples, 2))
    Y = np.logical_xor(X[:, 0], X[:, 1]).astype(int).reshape(-1, 1)
    return X.astype(np.float32), Y

NODES = [2, 4, 1]  # 2-input XOR, 1 hidden layer with 4 nodes, 1 output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--optimizer", default="adam") # momentum # SGD # adam
    parser.add_argument("--activation", default="relu")  # Works better for XOR
    parser.add_argument("--l_rate", default=0.01)
    parser.add_argument("--beta", default=0.9)
    parser.add_argument("--epochs", default=100)
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
    dnn = DeepNeuralNetwork(nodes=NODES, activation=args.activation)
    dnn.Initialization()
    dnn.Train(
        x_train, y_train, x_test, y_test,
        batch_size=int(args.batch_size),
        optimizer=args.optimizer,
        learning_rate=float(args.l_rate),
        beta=float(args.beta),
        epochs=int(args.epochs)
    )

    # Evaluate
    preds = dnn.Predict(x_test)
    acc = np.mean(preds == y_test.flatten())
    print(f"Final test accuracy: {acc:.4f}")

    # Show predictions on test set
    for i in range(10):
        print(f"Input: {x_test[i]} | Predicted: {preds[i]} | Actual: {y_test[i][0]}")
