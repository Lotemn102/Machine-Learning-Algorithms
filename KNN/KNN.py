import pandas as pd
import numpy as np
from data_loader import DataLoader
from utils import TRAIN_PATH, TEST_PATH
import math
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from typing import List


class KNN:
    def __init__(self, k: int = 1, normalized: bool = True):
        self.x_train = None
        self.x_test = None
        self.K = k
        self.normalized = normalized

    def fit(self, x_train: pd.DataFrame) -> None:
        """
        Normalize data with mini-max method.

        :param x_train: The training data set.
        :return: None.
        """
        # Avoid manipulations on original data.
        x = x_train.copy()

        # Normalize all features based on min-max.
        if self.normalized:
            for feature in x:

                if feature == 'diagnosis' or feature == 'Unnamed: 32':
                    continue

                min_val = min(x[feature])
                max_val = max(x[feature])
                x[feature] = x[feature].apply(lambda data: (data - min_val)/(max_val - min_val))

        self.x_train = x.copy()

    def predict(self, sample: np.array) -> str:
        """
        Make a prediction for a sample.

        :param sample: Numpy array with the sample's feature values.
        :return: A class prediction.
        """
        # Calculate euclidean distances to all samples in the train set.
        distances = []

        for index, row in self.x_train.iterrows():
            row = row.to_numpy()

            if 'M' in row or 'B' in row:
                # M or B are on the first place in the array.
                row = np.delete(row, 0)

            distances.append([index, self.calc_euclidean_dist(row, sample)])

        # Sort the distances.
        distances.sort(key=lambda x: x[1])

        # Get max class out of K nearest neighbors.
        classes = []

        if self.x_test is not None and self.K > len(self.x_test):
            k = len(self.x_test)
        else:
            k = self.K

        for i in range(k):
            index = distances[i][0]
            sample_class = self.x_train.loc[index, 'diagnosis']
            classes.append(sample_class)

        unique_values, values_counter = np.unique(classes, return_counts=True)
        return unique_values[values_counter.argmax()]

    def get_k_predictions(self, sample: np.array) -> List:
        """
        Get K-closest samples to a given sample.

        :param sample: Numpy array with the sample features.
        :return: List of K closest samples.
        """
        # Calculate euclidean distances to all samples in the train set.
        distances = []

        for index, row in self.x_train.iterrows():
            row = row.to_numpy()

            if 'M' in row or 'B' in row:
                # M or B are on the first place in the array.
                row = np.delete(row, 0)

            distances.append([index, self.calc_euclidean_dist(row, sample)])

        # Sort the distances.
        distances.sort(key=lambda x: x[1])

        # Get max class out of K nearest neighbors.
        classes = []

        if self.x_test is not None and self.K > len(self.x_test):
            k = len(self.x_test)
        else:
            k = self.K

        for i in range(k):
            index = distances[i][0]
            sample_class = self.x_train.loc[index, 'diagnosis']
            classes.append(sample_class)

        return classes

    def get_accuracy(self, x_test: pd.DataFrame) -> float:
        """
        Calculate the accuracy for the test set.

        :param x_test: The test set.
        :return: Accuracy.
        """
        self.x_test = x_test
        x = x_test.copy()

        # Normalize the test data.
        if self.normalized:
            for feature in x:

                if feature == 'diagnosis' or feature == 'Unnamed: 32':
                    continue

                min_val = min(x[feature])
                max_val = max(x[feature])

                if max_val == min_val:
                    continue

                x[feature] = x[feature].apply(lambda data: (data - min_val) / (max_val - min_val))

        x_test = x.copy()

        num_samples = x_test.shape[0]
        num_correct = 0

        for i in range(num_samples):
            sample = x_test.iloc[[i]]
            y = sample['diagnosis'].to_numpy()[0]
            sample = sample.to_numpy()
            y_pred = self.predict(sample)

            if y == y_pred:
                num_correct += 1

        return num_correct / num_samples

    def get_loss(self, x_test: pd.DataFrame) -> float:
        """
        Get the loss value as defined in question 5.

        :param x_test: The test set.
        :return: Loss value.
        """
        self.x_test = x_test
        x = x_test.copy()

        # Normalize the test data.
        for feature in x:

            if feature == 'diagnosis' or feature == 'Unnamed: 32':
                continue

            min_val = min(x[feature])
            max_val = max(x[feature])
            x[feature] = x[feature].apply(lambda data: (data - min_val) / (max_val - min_val))

        x_test = x.copy()

        num_samples = x_test.shape[0]
        fp = 0
        fn = 0

        for i in range(num_samples):
            sample = x_test.iloc[[i]]
            y = sample['diagnosis'].to_numpy()[0]
            sample = sample.to_numpy()
            y_pred = self.predict(sample)

            if y == 'M' and y_pred == 'B':
                fn += 1

            if y == 'B' and y_pred == 'M':
                fp += 1

        return ((0.1 * fp) + fn) / num_samples

    @staticmethod
    def calc_euclidean_dist(x: np.array, y: np.array) -> float:
        """
        Calculate the euclidean distance between 2 vectors.

        :param x: First vector on length N.
        :param y: Second vector on length N.
        :return: Euclidean distance.
        """
        if math.isnan(x[-1]):
            x = np.delete(x, -1)
            y = np.delete(y, -1)

        if 'M' in y or 'B' in y:
            # M or B are on the first place in the array.
            y = np.delete(y, 0)

        x_square = np.sum(x ** 2, axis=0)
        y_square = np.sum(y ** 2, axis=0)
        xy = x.dot(y)
        dist = np.sqrt(x_square + - 2 * xy + y_square)
        return dist


def train_and_test_knn():
    print("Training has started...")
    dl = DataLoader(train_path=TRAIN_PATH, test_path=TEST_PATH)
    x_train, y_train = dl.load_train_data()
    x_test, y_test = dl.load_test_data()

    model = KNN(k=3)
    model.fit(x_train)
    acc = model.get_accuracy(x_test=x_test)
    loss = model.get_loss(x_test=x_test)
    print("Finished.")
    print("Accuracy is: " + str(acc) + ", loss is: " + str(loss))


def experiment(K_values: List):
    """
    In order to run the experiment replace the function call at the bottom with this function.
    """
    dl = DataLoader(train_path='train.csv', test_path='')
    x_train, y_train = dl.load_train_data()

    num_splits = 5
    accuracies = []

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=316496736)

    for k in K_values:
        K_accuracies = []

        for train_index, validation_index in kf.split(x_train):
            _x_train = x_train.iloc[train_index]
            _y_train = _x_train['diagnosis'].to_numpy()

            _x_valid = x_train.iloc[validation_index]
            _y_valid = _x_valid['diagnosis'].to_numpy()

            model = KNN(k=k)
            model.fit(x_train=_x_valid)
            acc = model.get_accuracy(x_test=_x_valid)
            K_accuracies.append(acc)

        average_acc = sum(K_accuracies) / num_splits
        accuracies.append(average_acc)

    # Draw graph.
    x = K_values
    y = accuracies
    plt.plot(x, y)
    plt.xlabel('K values')
    plt.ylabel('Accuracy')
    plt.savefig('knn_experiment/experiment.png')


if __name__ == "__main__":
    # In order to train the model and test it.
    train_and_test_knn()

    # In order to tweak your model hyper-parameter of K, and print a graph describing accuracy per K value.
    #K_values = [3, 8, 30, 50, 100]
    #experiment(K_values)

