import numpy as np
from sklearn.datasets import fetch_openml
from DeepNeuralNetwork import DeepNeuralNetwork
import argparse
import matplotlib.pyplot as plt

def one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]

NODES = [784, 128, 128, 10]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--optimizer", default="adam") # relu # adam # SGD
    parser.add_argument("--activation", default="tanh") # sigmoid or tanh
    parser.add_argument("--l_rate", default=0.01)
    parser.add_argument("--beta", default=0.9)
    parser.add_argument("--epochs", default=10)
    args = parser.parse_args()

    print("Loading data...")
    mnist_data = fetch_openml("mnist_784", as_frame=False)
    x = mnist_data["data"]
    y = mnist_data["target"]

    print("Preprocessing data...")
    x = x.astype("float32") / 255.0
    y = y.astype("int32")

    # One-hot encode
    num_classes = 10
    y_new = one_hot(y, num_classes)

    # Split
    train_size = 60000
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y_new[:train_size], y_new[train_size:]

    # Shuffle training set
    shuffle_idx = np.random.permutation(train_size)
    x_train, y_train = x_train[shuffle_idx], y_train[shuffle_idx]

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
        epochs=args.epochs  # You can increase this
    )

    # Evaluate
    preds = dnn.Predict(x_test)
    acc = np.mean(preds == np.argmax(y_test, axis=1))
    print(f"Final test accuracy: {acc:.4f}")

    for i in range(5):
        image = x_test[i].reshape(28, 28)
        true_label = np.argmax(y_test[i])
        predicted_label = preds[i]

        plt.imshow(image, cmap="gray")
        plt.title(f"Predicted: {predicted_label} | Actual: {true_label}")
        plt.axis("off")
        plt.show()


