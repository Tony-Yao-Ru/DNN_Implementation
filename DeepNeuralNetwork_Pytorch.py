import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import time

TEMPLATE = "Epoch {}: {:.2f}s | train acc={:.2f} | train loss={:.4f} | test acc={:.2f} | test loss={:.4f}"

class DeepNeuralNetwork_Pytorch(nn.Module):
    def __init__(self, nodes, activation='sigmoid'):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        self.is_binary = False  # Will be determined in Train()

        for i in range(len(nodes) - 1):
            self.layers.append(nn.Linear(nodes[i], nodes[i + 1]))

        # Choose activation function
        if activation == 'relu':
            self.Activation = nn.ReLU()
        elif activation == 'tanh':
            self.Activation = nn.Tanh()
        else:
            self.Activation = nn.Sigmoid()

        self.to(self.device)

    def Initialization(self):
        pass  # Optional: Add custom initialization here

    def Train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=64,
              learning_rate=0.01, beta=0.9, beta1=0.9, beta2=0.999,
              epsilon=1e-8, optimizer="sgd"):

        # Input reshaping
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).reshape(-1, *x_train.shape[1:])
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).reshape(-1, *x_test.shape[1:])

        # Decide output type and loss based on last layer
        output_size = self.layers[-1].out_features
        if output_size == 1:
            loss_fn = nn.BCEWithLogitsLoss()
            self.is_binary = True
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        else:
            loss_fn = nn.CrossEntropyLoss()
            self.is_binary = False
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Create Datasets and Loaders
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        # Select optimizer
        if optimizer.lower() == 'sgd':
            opt = torch.optim.SGD(self.parameters(), lr=learning_rate)
        elif optimizer.lower() == "momentum":
            opt = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=beta)
        elif optimizer.lower() == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        start_time = time.time()

        # Training loop
        for epoch in range(epochs):
            train_acc, train_loss = self.train_loop(train_dataloader, loss_fn, opt)
            test_acc, test_loss = self.test_loop(test_dataloader, loss_fn)

            elapsed = time.time() - start_time
            print(TEMPLATE.format(epoch + 1, elapsed, train_acc, train_loss, test_acc, test_loss))

    def Forward_propagation(self, x):
        x = x.to(self.device)
        x = self.flatten(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.Activation(x)
        return x

    def Predict(self, input):
        x_tensor = torch.tensor(input, dtype=torch.float32).reshape(-1, *input.shape[1:]).to(self.device)
        self.eval()
        with torch.no_grad():
            output = self.Forward_propagation(x_tensor)
            if self.is_binary:
                preds = (output > 0).int().flatten()
            else:
                preds = torch.argmax(output, dim=1)
        return preds.cpu().numpy()

    def train_loop(self, dataloader, loss_fn, optimizer):
        size = len(dataloader.dataset)
        self.train()
        correct = 0
        total_loss = 0

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.Forward_propagation(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()

            if self.is_binary:
                pred_labels = (pred > 0).int()
                correct += (pred_labels.squeeze() == y.int()).sum().item()
            else:
                correct += (pred.argmax(1) == y).sum().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / size * 100
        return accuracy, avg_loss

    def test_loop(self, dataloader, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.Forward_propagation(X)
                test_loss += loss_fn(pred, y).item()

                if self.is_binary:
                    pred_labels = (pred > 0).int()
                    correct += (pred_labels.squeeze() == y.int()).sum().item()
                else:
                    correct += (pred.argmax(1) == y).sum().item()

        test_loss /= num_batches
        accuracy = correct / size * 100
        return accuracy, test_loss
