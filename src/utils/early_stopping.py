import torch
import numpy as np


class EarlyStopping:
    def __init__(self, path, patience=40, min_delta=0):
        self.path = path
        self.patience = patience
        self.min_delta = min_delta

        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False

    def __call__(self, score, model, epoch):
        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), self.path)
            self.counter = 0