import numpy as np
import time
import matplotlib.pyplot as plt

TEMPLATE = "Epoch {}: {:.2f}s | train acc={:.2f} | train loss={:.4f} | test acc={:.2f} | test loss={:.4f}"

class DeepNeuralNetwork_Numpy:
    def __init__(self, nodes, activation='sigmoid'):
        self.nodes = nodes
        self.layers = len(nodes)
        if activation == 'relu':
            self.Activation = self.relu  
        elif activation == 'tanh':
            self.Activation = self.tanh
        else:
            self.Activation = self.sigmoid

        self.cache = {}
        self.weighted_vectors = {}
        self.bias_vectors = {}
        self.gradients = {}
        self.t = 0  # timestep for Adam

    def Initialization(self):
        for i in range(self.layers - 1):
            self.weighted_vectors[f'w_{i}'] = np.random.randn(self.nodes[i+1], self.nodes[i]) * np.sqrt(1. / self.nodes[i])
            self.bias_vectors[f'b_{i+1}'] = np.zeros((self.nodes[i+1], 1))

    def Forward_propagation(self, input):
        if input.ndim == 1:
            input = input.reshape(-1, 1)
        else:
            input = input.T
        self.cache['z_0'] = input

        for i in range(1, self.layers):
            w = self.weighted_vectors[f"w_{i-1}"]
            b = self.bias_vectors[f"b_{i}"]
            z_prev = self.cache[f"z_{i-1}"]
            a = np.matmul(w, z_prev) + b

            if i == self.layers - 1:
                self.cache[f"z_{i}"] = self.sigmoid(a) if self.nodes[i] == 1 else self.softmax(a)
            else:
                self.cache[f"z_{i}"] = self.Activation(a)

        return self.cache[f"z_{self.layers - 1}"].T

    def Backward_propagation(self, Y):
        L = self.layers - 1
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        else:
            Y = Y.T

        m = Y.shape[1]
        dZ = self.cache[f"z_{L}"] - Y

        for i in reversed(range(1, self.layers)):
            A_prev = self.cache[f"z_{i-1}"]
            W = self.weighted_vectors[f"w_{i-1}"]

            dW = (1. / m) * np.dot(dZ, A_prev.T)
            db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)

            self.gradients[f"dW_{i-1}"] = dW
            self.gradients[f"db_{i}"] = db

            if i > 1:
                Z_prev = self.cache[f"z_{i-1}"]
                dA_prev = np.dot(W.T, dZ)
                dZ = dA_prev * self.Activation(Z_prev, derivative=True)

    def Optimizer(self, learning_rate=0.01, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if self.optimizer == "sgd":
            for i in range(1, self.layers):
                self.weighted_vectors[f"w_{i-1}"] -= learning_rate * self.gradients[f"dW_{i-1}"]
                self.bias_vectors[f"b_{i}"] -= learning_rate * self.gradients[f"db_{i}"]

        elif self.optimizer == "momentum":
            for i in range(1, self.layers):
                self.momentum_opt[f"w_{i-1}"] = beta * self.momentum_opt[f"w_{i-1}"] + (1 - beta) * self.gradients[f"dW_{i-1}"]
                self.momentum_opt[f"b_{i}"] = beta * self.momentum_opt[f"b_{i}"] + (1 - beta) * self.gradients[f"db_{i}"]
                self.weighted_vectors[f"w_{i-1}"] -= learning_rate * self.momentum_opt[f"w_{i-1}"]
                self.bias_vectors[f"b_{i}"] -= learning_rate * self.momentum_opt[f"b_{i}"]

        elif self.optimizer == "adam":
            self.t += 1
            for i in range(1, self.layers):
                self.adam_opt[f"m_w_{i-1}"] = beta1 * self.adam_opt[f"m_w_{i-1}"] + (1 - beta1) * self.gradients[f"dW_{i-1}"]
                self.adam_opt[f"m_b_{i}"] = beta1 * self.adam_opt[f"m_b_{i}"] + (1 - beta1) * self.gradients[f"db_{i}"]
                self.adam_opt[f"v_w_{i-1}"] = beta2 * self.adam_opt[f"v_w_{i-1}"] + (1 - beta2) * (self.gradients[f"dW_{i-1}"] ** 2)
                self.adam_opt[f"v_b_{i}"] = beta2 * self.adam_opt[f"v_b_{i}"] + (1 - beta2) * (self.gradients[f"db_{i}"] ** 2)

                m_w_hat = self.adam_opt[f"m_w_{i-1}"] / (1 - beta1 ** self.t)
                m_b_hat = self.adam_opt[f"m_b_{i}"] / (1 - beta1 ** self.t)
                v_w_hat = self.adam_opt[f"v_w_{i-1}"] / (1 - beta2 ** self.t)
                v_b_hat = self.adam_opt[f"v_b_{i}"] / (1 - beta2 ** self.t)

                self.weighted_vectors[f"w_{i-1}"] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                self.bias_vectors[f"b_{i}"] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    def Train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=64, learning_rate=0.01,
              beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, optimizer="sgd"):

        self.optimizer = optimizer
        self.t = 0
        self.is_binary = (self.nodes[-1] == 1)

        if self.optimizer == "momentum":
            self.momentum_opt = self.Initialize_momentum_opt()
        elif self.optimizer == "adam":
            self.adam_opt = self.Initialize_adam_opt()

        num_batches = -(-x_train.shape[0] // batch_size)
        start_time = time.time()

        self.loss_history = []

        # Real-time plot setup
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label="Training Loss", color="blue")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Live Training Loss")
        self.ax.grid(True)
        self.ax.legend()

        for epoch in range(epochs):
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for batch in range(num_batches):
                begin = batch * batch_size
                end = min(begin + batch_size, x_train.shape[0])
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]

                output = self.Forward_propagation(x)
                self.Backward_propagation(y)
                self.Optimizer(learning_rate=learning_rate, beta=beta, beta1=beta1, beta2=beta2, epsilon=epsilon)

            train_output = self.Forward_propagation(x_train)
            test_output = self.Forward_propagation(x_test)

            train_acc = self.accuracy(y_train, train_output)
            test_acc = self.accuracy(y_test, test_output)
            train_loss = self.cross_entropy_loss(y_train, train_output)
            test_loss = self.cross_entropy_loss(y_test, test_output)

            self.loss_history.append(train_loss)
            print(TEMPLATE.format(epoch + 1, time.time() - start_time, train_acc, train_loss, test_acc, test_loss))

            # Update plot
            self.line.set_xdata(np.arange(len(self.loss_history)))
            self.line.set_ydata(self.loss_history)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

        # Final plot update
        self.line.set_xdata(np.arange(len(self.loss_history)))
        self.line.set_ydata(self.loss_history)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.ioff()
        self.fig.savefig("training_curve.png")
        plt.show()

        return self.loss_history

    def Predict(self, X):
        output = self.Forward_propagation(X)
        if output.shape[1] == 1:
            return (output > 0.5).astype(int).flatten()
        else:
            return np.argmax(output, axis=1)

    def Initialize_momentum_opt(self):
        buffer = {}
        for i in range(self.layers - 1):
            buffer[f"w_{i}"] = np.zeros_like(self.weighted_vectors[f'w_{i}'])
            buffer[f"b_{i+1}"] = np.zeros_like(self.bias_vectors[f'b_{i+1}'])
        return buffer

    def Initialize_adam_opt(self):
        buffer = {}
        for i in range(self.layers - 1):
            buffer[f"m_w_{i}"] = np.zeros_like(self.weighted_vectors[f'w_{i}'])
            buffer[f"v_w_{i}"] = np.zeros_like(self.weighted_vectors[f'w_{i}'])
            buffer[f"m_b_{i+1}"] = np.zeros_like(self.bias_vectors[f'b_{i+1}'])
            buffer[f"v_b_{i+1}"] = np.zeros_like(self.bias_vectors[f'b_{i+1}'])
        return buffer

    def cross_entropy_loss(self, y, output):
        eps = 1e-12
        output = np.clip(output, eps, 1. - eps)

        if self.is_binary:
            return -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
        else:
            return -np.sum(y * np.log(output)) / y.shape[0]

    def accuracy(self, y, output):
        if output.shape[1] == 1:
            return np.mean((output > 0.5).astype(int).flatten() == y.flatten())
        return np.mean(np.argmax(y, axis=1) == np.argmax(output, axis=1))

    def relu(self, x, derivative=False):
        return np.where(x > 0, 1, 0) if derivative else np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig) if derivative else sig

    def tanh(self, x, derivative=False):
        if derivative:
            A = np.tanh(x)
            return 1 - A ** 2
        return np.tanh(x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)
import numpy as np
import time
import matplotlib.pyplot as plt

TEMPLATE = "Epoch {}: {:.2f}s | train acc={:.2f} | train loss={:.4f} | test acc={:.2f} | test loss={:.4f}"

class DeepNeuralNetwork_Numpy:
    def __init__(self, nodes, activation='sigmoid'):
        self.nodes = nodes
        self.layers = len(nodes)
        if activation == 'relu':
            self.Activation = self.relu  
        elif activation == 'tanh':
            self.Activation = self.tanh
        else:
            self.Activation = self.sigmoid

        self.cache = {}
        self.weighted_vectors = {}
        self.bias_vectors = {}
        self.gradients = {}
        self.t = 0  # timestep for Adam

    def Initialization(self):
        for i in range(self.layers - 1):
            self.weighted_vectors[f'w_{i}'] = np.random.randn(self.nodes[i+1], self.nodes[i]) * np.sqrt(1. / self.nodes[i])
            self.bias_vectors[f'b_{i+1}'] = np.zeros((self.nodes[i+1], 1))

    def Forward_propagation(self, input):
        if input.ndim == 1:
            input = input.reshape(-1, 1)
        else:
            input = input.T
        self.cache['z_0'] = input

        for i in range(1, self.layers):
            w = self.weighted_vectors[f"w_{i-1}"]
            b = self.bias_vectors[f"b_{i}"]
            z_prev = self.cache[f"z_{i-1}"]
            a = np.matmul(w, z_prev) + b

            if i == self.layers - 1:
                self.cache[f"z_{i}"] = self.sigmoid(a) if self.nodes[i] == 1 else self.softmax(a)
            else:
                self.cache[f"z_{i}"] = self.Activation(a)

        return self.cache[f"z_{self.layers - 1}"].T

    def Backward_propagation(self, Y):
        L = self.layers - 1
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        else:
            Y = Y.T

        m = Y.shape[1]
        dZ = self.cache[f"z_{L}"] - Y

        for i in reversed(range(1, self.layers)):
            A_prev = self.cache[f"z_{i-1}"]
            W = self.weighted_vectors[f"w_{i-1}"]

            dW = (1. / m) * np.dot(dZ, A_prev.T)
            db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)

            self.gradients[f"dW_{i-1}"] = dW
            self.gradients[f"db_{i}"] = db

            if i > 1:
                Z_prev = self.cache[f"z_{i-1}"]
                dA_prev = np.dot(W.T, dZ)
                dZ = dA_prev * self.Activation(Z_prev, derivative=True)

    def Optimizer(self, learning_rate=0.01, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if self.optimizer == "sgd":
            for i in range(1, self.layers):
                self.weighted_vectors[f"w_{i-1}"] -= learning_rate * self.gradients[f"dW_{i-1}"]
                self.bias_vectors[f"b_{i}"] -= learning_rate * self.gradients[f"db_{i}"]

        elif self.optimizer == "momentum":
            for i in range(1, self.layers):
                self.momentum_opt[f"w_{i-1}"] = beta * self.momentum_opt[f"w_{i-1}"] + (1 - beta) * self.gradients[f"dW_{i-1}"]
                self.momentum_opt[f"b_{i}"] = beta * self.momentum_opt[f"b_{i}"] + (1 - beta) * self.gradients[f"db_{i}"]
                self.weighted_vectors[f"w_{i-1}"] -= learning_rate * self.momentum_opt[f"w_{i-1}"]
                self.bias_vectors[f"b_{i}"] -= learning_rate * self.momentum_opt[f"b_{i}"]

        elif self.optimizer == "adam":
            self.t += 1
            for i in range(1, self.layers):
                self.adam_opt[f"m_w_{i-1}"] = beta1 * self.adam_opt[f"m_w_{i-1}"] + (1 - beta1) * self.gradients[f"dW_{i-1}"]
                self.adam_opt[f"m_b_{i}"] = beta1 * self.adam_opt[f"m_b_{i}"] + (1 - beta1) * self.gradients[f"db_{i}"]
                self.adam_opt[f"v_w_{i-1}"] = beta2 * self.adam_opt[f"v_w_{i-1}"] + (1 - beta2) * (self.gradients[f"dW_{i-1}"] ** 2)
                self.adam_opt[f"v_b_{i}"] = beta2 * self.adam_opt[f"v_b_{i}"] + (1 - beta2) * (self.gradients[f"db_{i}"] ** 2)

                m_w_hat = self.adam_opt[f"m_w_{i-1}"] / (1 - beta1 ** self.t)
                m_b_hat = self.adam_opt[f"m_b_{i}"] / (1 - beta1 ** self.t)
                v_w_hat = self.adam_opt[f"v_w_{i-1}"] / (1 - beta2 ** self.t)
                v_b_hat = self.adam_opt[f"v_b_{i}"] / (1 - beta2 ** self.t)

                self.weighted_vectors[f"w_{i-1}"] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                self.bias_vectors[f"b_{i}"] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    def Train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=64, learning_rate=0.01,
              beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, optimizer="sgd"):

        self.optimizer = optimizer
        self.t = 0
        self.is_binary = (self.nodes[-1] == 1)

        if self.optimizer == "momentum":
            self.momentum_opt = self.Initialize_momentum_opt()
        elif self.optimizer == "adam":
            self.adam_opt = self.Initialize_adam_opt()

        num_batches = -(-x_train.shape[0] // batch_size)
        start_time = time.time()

        self.loss_history = []

        # Real-time plot setup
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label="Training Loss", color="blue")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training Loss")
        self.ax.grid(True)
        self.ax.legend()

        for epoch in range(epochs):
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for batch in range(num_batches):
                begin = batch * batch_size
                end = min(begin + batch_size, x_train.shape[0])
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]

                output = self.Forward_propagation(x)
                self.Backward_propagation(y)
                self.Optimizer(learning_rate=learning_rate, beta=beta, beta1=beta1, beta2=beta2, epsilon=epsilon)

            train_output = self.Forward_propagation(x_train)
            test_output = self.Forward_propagation(x_test)

            train_acc = self.accuracy(y_train, train_output)
            test_acc = self.accuracy(y_test, test_output)
            train_loss = self.cross_entropy_loss(y_train, train_output)
            test_loss = self.cross_entropy_loss(y_test, test_output)

            self.loss_history.append(train_loss)
            print(TEMPLATE.format(epoch + 1, time.time() - start_time, train_acc, train_loss, test_acc, test_loss))

            # Update plot
            self.line.set_xdata(np.arange(len(self.loss_history)))
            self.line.set_ydata(self.loss_history)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

        # Final plot update
        self.line.set_xdata(np.arange(len(self.loss_history)))
        self.line.set_ydata(self.loss_history)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.ioff()
        plt.show()

        return self.loss_history

    def Predict(self, X):
        output = self.Forward_propagation(X)
        if output.shape[1] == 1:
            return (output > 0.5).astype(int).flatten()
        else:
            return np.argmax(output, axis=1)

    def Initialize_momentum_opt(self):
        buffer = {}
        for i in range(self.layers - 1):
            buffer[f"w_{i}"] = np.zeros_like(self.weighted_vectors[f'w_{i}'])
            buffer[f"b_{i+1}"] = np.zeros_like(self.bias_vectors[f'b_{i+1}'])
        return buffer

    def Initialize_adam_opt(self):
        buffer = {}
        for i in range(self.layers - 1):
            buffer[f"m_w_{i}"] = np.zeros_like(self.weighted_vectors[f'w_{i}'])
            buffer[f"v_w_{i}"] = np.zeros_like(self.weighted_vectors[f'w_{i}'])
            buffer[f"m_b_{i+1}"] = np.zeros_like(self.bias_vectors[f'b_{i+1}'])
            buffer[f"v_b_{i+1}"] = np.zeros_like(self.bias_vectors[f'b_{i+1}'])
        return buffer

    def cross_entropy_loss(self, y, output):
        eps = 1e-12
        output = np.clip(output, eps, 1. - eps)

        if self.is_binary:
            return -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
        else:
            return -np.sum(y * np.log(output)) / y.shape[0]

    def accuracy(self, y, output):
        if output.shape[1] == 1:
            return np.mean((output > 0.5).astype(int).flatten() == y.flatten())
        return np.mean(np.argmax(y, axis=1) == np.argmax(output, axis=1))

    def relu(self, x, derivative=False):
        return np.where(x > 0, 1, 0) if derivative else np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig) if derivative else sig

    def tanh(self, x, derivative=False):
        if derivative:
            A = np.tanh(x)
            return 1 - A ** 2
        return np.tanh(x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)
