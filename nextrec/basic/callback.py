"""
EarlyStopper definitions

Date: create on 27/10/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import copy


class EarlyStopper(object):
    def __init__(self, patience: int = 20, mode: str = "max"):
        self.patience = patience
        self.trial_counter = 0
        self.best_metrics = 0
        self.best_weights = None
        self.mode = mode

    def stop_training(self, val_metrics, weights):
        if self.mode == "max":
            if val_metrics > self.best_metrics:
                self.best_metrics = val_metrics
                self.trial_counter = 0
                self.best_weights = copy.deepcopy(weights)
        elif self.mode == "min":
            if val_metrics < self.best_metrics:
                self.best_metrics = val_metrics
                self.trial_counter = 0
                self.best_weights = copy.deepcopy(weights)
            return False
        elif self.trial_counter + 1 < self.patience:
            self.trial_counter += 1
            return False
        else:
            return True
