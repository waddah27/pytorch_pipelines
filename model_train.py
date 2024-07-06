
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data_loader import DataLoaders
from tqdm import tqdm

class model_trainer:
    """
    Model wrapper for training and testing a PyTorch model.

    Args:
        model (nn.Module): PyTorch model to be trained or tested.
        loader (DataLoader): DataLoader object for loading and preprocessing data.
        n_epochs (int, optional): Number of epochs to train the model. Defaults to 1.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        is_rnn (bool, optional): Flag indicating if the model is a recurrent neural network.
            Defaults to False.
        device (str, optional): Device to be used for training. Defaults to "cpu".
    """
    def __init__(self, model: nn.Module, model_name: str, n_epochs: int = 1, lr: float = 0.001,
                 is_rnn: bool = False, device: str = "cpu"):
        self.model = model
        self.ckpt_filename = f"ckpt_{model_name}.pth.tar"
        self.n_epochs = n_epochs
        self.is_rnn = is_rnn
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.accuracy = 0

    def train(self, loaders: DataLoaders, load_checkpoint: bool = False):
        """
        Trains the model for a specified number of epochs.
        """
        train_loader = loaders["train"]
        test_loader = loaders["test"]
        if load_checkpoint:
            try:
                self.load_checkpoint()
            except:
                print("Failed to load checkpoint. Starting from scratch.")

        self.model = self.model.to(self.device)
        for epoch in tqdm(range(self.n_epochs)):
            for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

                if self.is_rnn:
                    data = data.squeeze(1)
                data = data.to(self.device)
                targets = targets.to(self.device)
                scores = self.model(data)
                loss = self.criterion(scores, targets)
                tqdm(desc=f"Epoch {epoch+1}/{self.n_epochs}, batch {batch_idx+1}/{len(train_loader)}, loss = {loss.item():.4f}")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            test_acc = self.check_accuracy(test_loader)
            train_acc = self.check_accuracy(train_loader)
            if test_acc > self.accuracy:
                self.accuracy = test_acc
                self.save_checkpoint()
        return self

    def predict(self, x):
        """
        Predicts the class labels for a given input data.

        Args:
            data (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Predicted class labels.
        """
        self.model.eval()
        with torch.no_grad():
            if self.is_rnn:
                x = x.squeeze(1)
            x = x.to(self.device)
            scores = self.model(x)
            _, predictions = scores.max(1)
            return predictions

    def check_accuracy(self, loader):
        """
        Checks the accuracy of the model on the training or test data.

        Returns:
            float: Accuracy of the model.
        """
        if loader.dataset.train:
            print("checking accuracy on training data")
        else:
            print("checking accuracy on test data")

        num_correct = 0
        num_samples = 0
        accuracy = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                y = y.to(self.device)
                preds = self.predict(x)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
        accuracy = float(num_correct) / num_samples
        print(f"Got {num_correct} / {num_samples} with accuracy {accuracy*100:.2f}")
        self.model.train()
        return accuracy

    def save_checkpoint(self):
        """
        Saves the model state and optimizer state to a checkpoint file.

        """

        ckpt = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.n_epochs,
            "accuracy": self.accuracy,
        }
        print("=> Saving checkpoint")
        torch.save(ckpt, self.ckpt_filename)

    def load_checkpoint(self):
        print("=> Loading checkpoint")
        ckpt = torch.load(self.ckpt_filename, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.n_epochs = ckpt["epoch"]
        self.accuracy = ckpt["accuracy"]

