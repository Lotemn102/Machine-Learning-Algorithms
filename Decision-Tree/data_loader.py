import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load_train_data(self) -> (np.array, np.array):
        x_train = pd.read_csv(self.train_path, sep=',')
        y_train = x_train['diagnosis'].to_numpy()
        return x_train, y_train

    def load_test_data(self) -> (np.array, np.array):
        x_test = pd.read_csv(self.test_path, sep=',')
        y_test = x_test['diagnosis'].to_numpy()
        return x_test, y_test

