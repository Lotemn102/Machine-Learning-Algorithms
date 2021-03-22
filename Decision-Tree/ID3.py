from tree_node import TreeNode
from data_loader import DataLoader
from utils import TRAIN_PATH, TEST_PATH
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from typing import List


class ID3:
    def __init__(self, pruning_M: int = 0):
        self.pruning_M = pruning_M

    def fit(self, x_train: pd.DataFrame, y_train: np.array) -> TreeNode:
        """
        Create a decision tree for the training data, using the max information gain heuristic.
        :param x_train: Pandas data frame with the data (features) to train the model on. The array length is N*M where
                N is the number of samples in the dataset, and M is the number of features.
        :param y_train: Numpy array with the labels of the data. Length is N.
        :return: A decision tree.
        """
        tree = TreeNode('ROOT')
        self.build_tree(x_train=x_train, y_train=y_train, tree=tree)
        return tree

    def predict(self, sample: pd.DataFrame, tree: TreeNode) -> str:
        """
        Get a predicted class for a sample.

        :param sample: Pandas data frame with the features of the sample.
        :param tree: The decision tree the model created.
        :return: Predicted class.
        """
        if tree.data == 'M' or tree.data == 'B':
            return tree.data

        if tree.data == 'ROOT':
            if len(tree.children) == 0:
                return 'No nodes on tree.'
            else:
                # Skip the root.
                tree = tree.children[0]

                # If the original tree had only ROOT and one son.
                if len(tree.children) == 0:
                    return tree.data

        feature_name = tree.data[0]
        threshold = tree.data[1]

        if sample[feature_name].to_numpy()[0] >= float(threshold):
            tree = tree.children[0]
        else:
            tree = tree.children[1]

        return self.predict(sample=sample, tree=tree)

    def get_accuracy(self, tree: TreeNode, x_test: pd.DataFrame) -> float:
        """
        Calculate the accuracy of the decision tree over a test set.

        :param tree: The decision tree the model created.
        :param x_test: Pandas data frame with the features of the test samples.
        :return: Accuracy.
        """
        num_samples = x_test.shape[0]
        num_correct = 0

        for i in range(num_samples):
            sample = x_test.iloc[[i]]
            y = sample['diagnosis'].to_numpy()[0]
            y_pred = self.predict(sample, tree)

            if y == y_pred:
                num_correct += 1

        return num_correct / num_samples

    def get_loss(self, tree: TreeNode, x_test: pd.DataFrame) -> float:
        """
        Get the loss value as defined in question 4.

        :param tree: The decision tree the model created.
        :param x_test: The test set.
        :return: Loss value.
        """
        num_samples = x_test.shape[0]
        fp = 0
        fn = 0

        for i in range(num_samples):
            sample = x_test.iloc[[i]]
            y = sample['diagnosis'].to_numpy()[0]
            y_pred = self.predict(sample, tree)

            if y == 'M' and y_pred == 'B':
                fn += 1

            if y == 'B' and y_pred == 'M':
                fp += 1

        return ((0.1 * fp) + fn) / num_samples

    def build_tree(self, x_train: pd.DataFrame, y_train: np.array, tree: TreeNode) -> None:
        """
        Build decision tree with ID3 feature choosing algorithm.

        :param x_train: Pandas data frame with the data (features) to train the model on. The array length is N*M where
                N is the number of samples in the dataset, and M is the number of features.
        :param y_train: Numpy array with the labels of the data. Length is N.
        :param tree: A root node.
        :return: None.
        """
        # Create new leaf with the class.
        if x_train.shape[0] == 1 or y_train.shape[0] <= self.pruning_M or self.are_all_labels_equal(y_train):
            new_node = TreeNode(self.find_major_class(y_train))
            tree.add_child(new_node)
            return

        # Find best feature to continue with, and best threhold to split the data with it.
        f_name, best_threshold = self.get_feature_with_max_info_gain(x_train, y_train)

        # Add another layer to the tree representing the current f, add thresh for prediction later.
        new_node = TreeNode((f_name, str(best_threshold)))
        tree.add_child(new_node)

        samples_great, labels_great, samples_small, labels_small = self.split_by_threshold(data=x_train,
                                                                                           threshold=best_threshold,
                                                                                           feature_name=f_name)
        if samples_great.shape[0] == 1:
            # If there is only one sample, no need to do recursion call.
            node = TreeNode(labels_great[0])
            new_node.add_child(node)
        else:
            self.build_tree(samples_great, labels_great, new_node)

        if samples_small.shape[0] == 1:
            # If there is only one sample, no need to do recursion call.
            node = TreeNode(labels_small[0])
            new_node.add_child(node)
        else:
            self.build_tree(samples_small, labels_small, new_node)

        return

    @staticmethod
    def split_by_threshold(data: pd.DataFrame, threshold: float, feature_name: str) ->\
            (pd.DataFrame, np.array, pd.DataFrame, np.array):
        """
        For a given feature and a threshold, create new 2 subsets of the data: one where all samples with feature value
        greater than threshold, and the other where all samples with feature value smaller the threshold.
        :param data: All samples in some node.
        :param threshold: A float.
        :param feature_name: A string.
        :return: A tuple of (all samples greater than threshold, their labels, all samples smaller than threshold,
         their labels)
        """
        data_great = data.loc[data[feature_name] >= threshold]  # >= is because of the segel's comment in ex. 1.1.
        labels_great = data_great['diagnosis'].to_numpy()

        data_small = data.loc[data[feature_name] < threshold]
        labels_small = data_small['diagnosis'].to_numpy()

        return data_great, labels_great, data_small, labels_small

    @staticmethod
    def find_major_class(y: np.array) -> str:
        """
        Find the class that has maximum occurrences in the labels.
        :param y: Numpy array of length N, where N is the number of samples.
        :return: A class ('M' or 'B').
        """
        unique_values, values_counter = np.unique(y, return_counts=True)
        return unique_values[values_counter.argmax()]

    @staticmethod
    def are_all_labels_equal(y: np.array) -> bool:
        """
        Check if all labels in a numpy array are the same.
        :param y: Numpy array of length N, where N is the number of samples.
        :return: Boolean value.
        """
        res = np.all(y == y[0])
        return res

    def get_feature_with_max_info_gain(self, x: pd.DataFrame, y: np.array) -> (str, float):
        """
        Get the feature which maximize the information gain.
        :param x: Pandas data frame of length N * M, where N is number of samples and M is number of features.
        :param y: Labels of the data. Length N.
        :return: Feature which maximize the information gain, and best threshold to split data with this feature.
        """
        max_info_gain = -1
        max_feature_name = ''
        max_threshold = 0
        features_names = x.columns.to_list()

        if 'diagnosis' in features_names:
            features_names.remove('diagnosis')

        if 'Unnamed: 32' in features_names:
            features_names.remove('Unnamed: 32')

        for name in features_names:
            best_threshold, IG = self.find_best_threshold(x=x, f_name=name)

            if IG >= max_info_gain: # >= is because segel's comment in ex. 1.1 (take max index feature).
                max_info_gain = IG
                max_feature_name = name
                max_threshold = best_threshold

        return max_feature_name, max_threshold

    @staticmethod
    def calc_entropy(labels: np.array) -> float:
        """
        Calculate entropy for a vector of labels.
        :param labels: Numpy array of length N, where N is number of samples.
        :return: Entropy.
        """
        if len(labels) == 0:
            return 0

        classes, counts = np.unique(ar=labels, return_counts=True)
        p = np.true_divide(counts, len(labels))
        entropy = np.sum(np.multiply(p, np.log2(p)))
        entropy *= -1
        return float(entropy)

    def find_best_threshold(self, x: pd.DataFrame, f_name: str) -> (float, float):
        """
        Given a feature, find best value to split the samples in some node according to it.
        :param x: The data.
        :param f_name: Feature to find a threshold in.
        :return: Best threshold and it's info gain.
        """
        min_info_gain = float('inf')
        feature_values = x[f_name].to_numpy()
        labels = x['diagnosis'].to_numpy()
        entropy = self.calc_entropy(labels)
        labels = [label for _, label in sorted(zip(feature_values, labels))]
        feature_values = np.sort(feature_values, axis=None)
        best_threshold = 0
        num_samples = feature_values.shape[0]

        if num_samples == 2:
            return ((feature_values[0] + feature_values[1]) / 2.0), 0

        for i in range(1, num_samples):
            if feature_values[i - 1] != feature_values[i]:
                info_gain = i * self.calc_entropy(labels[0: i]) + (num_samples - i) * \
                            self.calc_entropy(labels[i:])

                if info_gain < min_info_gain:
                    min_info_gain = info_gain
                    best_threshold = (feature_values[i] + feature_values[i - 1]) / 2.0

        if min_info_gain == float('inf'):
            # If all values in f are equal.
            min_info_gain = -1
            return 0, min_info_gain

        best_IG = entropy - min_info_gain / num_samples
        return best_threshold, best_IG


def train_and_test_tree():
    print("Training has started...")
    dl = DataLoader(train_path=TRAIN_PATH, test_path=TEST_PATH)
    x_train, y_train = dl.load_train_data()
    x_test, y_test = dl.load_test_data()

    model = ID3(pruning_M=0)
    decision_tree = model.fit(x_train, y_train)
    acc = model.get_accuracy(tree=decision_tree, x_test=x_test)
    print("Finished.")
    print("Accuracy is: " + str(acc))


def experiment(M_values: List):
    """
    In order to run the experiment call this function on the main section.
    """
    dl = DataLoader(train_path=TRAIN_PATH, test_path=TEST_PATH)
    x_train, y_train = dl.load_train_data()
    num_splits = 5
    accuracies = []

    for M in M_values:
        m_accuracies = []
        kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

        for train_index, validation_index in kf.split(x_train):
            _x_train = x_train.iloc[train_index]
            _y_train = _x_train['diagnosis'].to_numpy()

            _x_valid = x_train.iloc[validation_index]
            _y_valid = _x_valid['diagnosis'].to_numpy()

            model = ID3(pruning_M=M)
            decision_tree = model.fit(_x_train, _y_train)
            acc = model.get_accuracy(tree=decision_tree, x_test=_x_valid)
            m_accuracies.append(acc)

        average_acc = sum(m_accuracies) / num_splits
        accuracies.append(average_acc)

    # Draw graph.
    x = M_values
    y = accuracies
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    plt.plot(x, y, color=palette(0), linewidth=1, alpha=0.9)
    plt.xlabel('M values')
    plt.ylabel('Accuracy')
    plt.title('Accuracy as function of M pruning values')
    plt.savefig('experiment_prune.png')


if __name__ == '__main__':
    # In order to train the model and test it.
    train_and_test_tree()

    # In order to tweak your model hyper-parameter of M, and print a graph describing accuracy per K value.
    # M_values = [5, 10, 20, 30, 50]
    # experiment(M_values)


