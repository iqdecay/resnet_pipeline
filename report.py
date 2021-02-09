import logging
import os
import pickle


class Report(object):
    def __init__(self, directory, archi, lr, n_epochs, train_losses, val_losses,
                 train_time, accuracy,
                 inference_time):
        self.directory = directory
        self.architecture = archi
        self.lr = lr
        self.n_epochs = n_epochs
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_time = train_time
        self.accuracy = accuracy
        self.inference_time = inference_time

    def save(self):
        path = os.path.join(self.directory, "report.pkl")
        with open(path, "wb") as file:
            pickle.dump(self, file)
            logging.info(
                f"Saved training and inference information in {path}")
