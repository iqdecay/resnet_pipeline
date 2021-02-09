import logging
import os
import time

import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

from constants import MODEL_DIR, GPU

from minicifar import minicifar_train, minicifar_test, train_sampler, \
    valid_sampler


class Pipeline(object):
    def __init__(self, model: nn.Module, lr: float, n_epochs: int, name: str):
        self.name = name
        self.directory = os.path.join(MODEL_DIR, name)
        self.n_epochs = n_epochs
        self.model = model
        self.lr = lr
        self.train_loader = DataLoader(minicifar_train, batch_size=32,
                                       sampler=train_sampler)
        self.valid_loader = DataLoader(minicifar_train, batch_size=32,
                                       sampler=valid_sampler)
        self.test_loader = DataLoader(minicifar_test, batch_size=32)

    def train(self, tol=1e-4, n_iter_no_change=10, momentum=0.9):
        logging.info(
            f"Training model {self.name} on {self.n_epochs} epochs with lr={self.lr}")
        net = self.model
        # Move model to GPU
        net.to(GPU)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=momentum)
        training_losses = []
        validation_losses = []
        best_validation_loss = 1e10
        no_improvement_count = 0
        start_time = time.time()

        for epoch in range(self.n_epochs):
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch + 1}/{self.n_epochs}")
            current_loss = 0.0
            n_batch = 0
            # Training
            for i, data in enumerate(self.train_loader, 0):
                X, y = data
                X = X.to(GPU)
                y = y.to(GPU)
                optimizer.zero_grad()

                # Forward then backward pass
                out = net(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                n_batch += 1
                current_loss += loss.item()
            training_losses.append(current_loss / n_batch)
            # Validation
            current_loss = 0.0
            n_batch = 0
            with torch.set_grad_enabled(False):
                for i, data in enumerate(self.valid_loader, 0):
                    X, y = data
                    X = X.to(GPU)
                    y = y.to(GPU)
                    out = net(X)
                    loss = criterion(out, y)
                    n_batch += 1
                    current_loss += loss.item()
            validation_loss = current_loss / n_batch
            validation_losses.append(validation_loss)
            # Early stopping
            if validation_loss > (best_validation_loss - tol):
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
            if no_improvement_count > n_iter_no_change:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break
        duration = time.time() - start_time
        logging.info(
            f"Finished training {self.name} on {self.n_epochs} epochs")

        net.cpu()
        self.model = net
        return training_losses, validation_losses, duration

    def test(self):
        correct = 0
        total = 0
        self.model.to(GPU)
        start_time = time.time()
        logging.info(
            f"Running inference on model {self.name}")
        with torch.no_grad():
            for data in self.test_loader:
                X, y = data
                X = X.to(GPU)
                y = y.to(GPU)
                out = self.model(X)
                _, predicted = torch.max(out.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = 100 * correct / total
        duration = time.time() - start_time
        self.model.cpu()
        logging.info(
            f"Accuracy of the network on the {total} test images: {accuracy}%")
        return accuracy, duration

    def run(self, overwrite=True):
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        # TODO : use train and val_losses and so on
        train_losses, val_losses, duration = self.train()
        if overwrite:
            path = os.path.join(self.directory, self.name + ".pth")
            torch.save(self.model.state_dict(), path)
            logging.info(f"Saved model under {path}")
        # TODO : use the accuracy
        accuracy, duration = self.test()
        del self.model
