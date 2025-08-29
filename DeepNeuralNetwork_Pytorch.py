import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt

TEMPLATE = "Epoch {}: {:.2f}s | train acc={:.2f} | train loss={:.4f} | test acc={:.2f} | test loss={:.4f}"

class DeepNeuralNetwork_Pytorch(nn.Module):
    def __init__(self, nodes, activation='sigmoid'):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(self.device)

        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        self.is_binary = False

        for i in range(len(nodes) - 1):
            self.layers.append(nn.Linear(nodes[i], nodes[i + 1]))

        if activation == 'relu':
            self.Activation = nn.ReLU()
        elif activation == 'tanh':
            self.Activation = nn.Tanh()
        else:
            self.Activation = nn.Sigmoid()

        self.to(self.device)
    
    def Initialization(self):
        pass

    def Train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=64,
          learning_rate=0.01, beta=0.9, beta1=0.9, beta2=0.999,
          epsilon=1e-8, optimizer="sgd"):

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).reshape(-1, *x_train.shape[1:])
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).reshape(-1, *x_test.shape[1:])

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

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        if optimizer.lower() == 'sgd':
            opt = torch.optim.SGD(self.parameters(), lr=learning_rate)
        elif optimizer.lower() == "momentum":
            opt = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=beta)
        elif optimizer.lower() == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        train_loss_list = []
        epochs_list = []

        start_time = time.time()

        plt.ion()
        fig, ax = plt.subplots(figsize=(8,5))
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()

        for epoch in range(epochs):
            _, train_loss = self.train_loop(train_dataloader, loss_fn, opt)

            epochs_list.append(epoch + 1)
            train_loss_list.append(train_loss)

            elapsed = time.time() - start_time
            # Print train loss only (you can keep test metrics printing if you want)
            print(f"Epoch {epoch+1}: {elapsed:.2f}s | train loss={train_loss:.4f}")

            ax.cla()
            ax.plot(epochs_list, train_loss_list, label="Train Loss", color='red')
            ax.set_title("Training Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.pause(0.1)

        plt.ioff()
        plt.show()


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
