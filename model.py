import numpy as np
import pandas
import pandas as pd


class DecisionTree:
    def __init__(self, data, number_of_folds, max_depth, min_size, cost_function='GINI'):
        self.max_depth = max_depth
        self.min_size = min_size
        self.cost_function = cost_function

        self.dataset = None
        self.cross_validation_split(data, number_of_folds)

    def run(self):
        accuracies = []
        for fold_index, fold in enumerate(self.dataset):
            test_set = fold

            mask = np.ones(self.dataset.shape[0], bool)
            mask[fold_index] = False
            train_set = self.dataset[mask].reshape(-1, self.dataset.shape[2])

            predictions = self.run_decision_tree(train_set, test_set)
            labels = test_set[:, -1]

            accuracy = ((predictions == labels).sum() / labels.shape[0]) * 100
            accuracies.append(accuracy)
        return accuracies

    def run_decision_tree(self, train_set, test_set):
        tree = self.build_tree(train_set)
        predictions = []
        for row in test_set:
            predictions.append(self.predict(tree, row))
        return np.array(predictions)

    def cross_validation_split(self, data, number_of_folds):
        if not isinstance(data, np.ndarray):
            if isinstance(data, pandas.DataFrame):
                data = data.to_numpy()
            elif isinstance(data, list):
                data = np.ndarray(data)
            else:
                print("Invalid type")
                exit(-1)
        data_length = data.shape[0]
        fold_size = data_length // number_of_folds
        for i in range(number_of_folds):
            start = i * fold_size
            fold = data[start:start + fold_size, :]
            fold = fold[np.newaxis, ...]  # add additional dimension, otherwise after vstack we get the initial data
            if self.dataset is None:
                self.dataset = fold
            else:
                self.dataset = np.vstack([self.dataset, fold])

    def cost(self, groups):
        if self.cost_function == "GINI":
            return self.gini_index(groups)
        elif self.cost_function == "ENTROPY":
            return self.entropy(groups)
        elif self.cost_function == "CLASSIFICATION ERROR":
            return self.classification_error(groups)
        else:
            print("ERROR: INVALID COST FUNCTION")
            exit()

    @staticmethod
    def gini_index(groups):
        # we remove last element from .shape when counting number of instances because we only care about labels in
        # each row
        number_of_instances = np.prod(groups.shape[:-1])

        gini = 0
        for group in groups:
            group_size = group.shape[0]
            if group_size == 0:
                continue

            group_classes = group[:, -1]
            _, counts = np.unique(group_classes, return_counts=True)
            p = counts / group_size
            sigma = np.sum(p * p)
            group_weight = group_size / number_of_instances
            gini += (1 - sigma) * group_weight

        return gini

    @staticmethod
    def entropy(groups):
        number_of_instances = np.prod(groups.shape[:-1])

        entropy = 0
        for group in groups:
            group_size = group.shape[0]
            if group_size == 0:
                continue

            group_classes = group[:, -1]
            _, counts = np.unique(group_classes, return_counts=True)
            p = counts / group_size
            sigma = np.sum(p * np.log2(p))
            group_weight = group_size / number_of_instances
            entropy += (- sigma) * group_weight

        return entropy

    @staticmethod
    def classification_error(groups):
        number_of_instances = np.prod(groups.shape[:-1])

        error = 0
        for group in groups:
            group_size = group.shape[0]
            if group_size == 0:
                continue

            group_classes = group[:, -1]
            _, counts = np.unique(group_classes, return_counts=True)
            p = counts / group_size
            max_p = np.max(p)
            group_weight = group_size / number_of_instances
            error += (1 - max_p) * group_weight

        return error

    @staticmethod
    def test_split(feature_index, value, data):
        left = data[data[:, feature_index] < value]
        right = data[data[:, feature_index] >= value]
        return np.array([left, right], dtype=object)

    def get_split(self, data):
        best_index, best_value, best_score, best_groups = float('inf'), float('inf'), float('inf'), None
        for feature_index in range(data.shape[1] - 1):  # except last column
            for row in data:
                groups = self.test_split(feature_index, row[feature_index], data)
                cost_score = self.cost(groups)
                if cost_score < best_score:
                    best_index, best_value, best_score, best_groups = feature_index, row[feature_index], cost_score, groups
        return {
            'index': best_index,
            'value': best_value,
            'groups': best_groups
        }

    @staticmethod
    def to_terminal(group):
        outcomes = group[:, -1]
        unique_outcomes, counts = np.unique(outcomes, return_counts=True)
        most_frequent_index = np.argmax(counts)
        return unique_outcomes[most_frequent_index]

    def split(self, node, depth):
        left, right = node['groups']
        del (node['groups'])

        if left.shape[0] == 0 or right.shape[0] == 0:
            node['left'] = node['right'] = self.to_terminal(np.vstack((left, right)))
            return
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)

        if left.shape[0] <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], depth + 1)

        if right.shape[0] <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], depth + 1)

    def build_tree(self, train_set):
        root = self.get_split(train_set)
        self.split(root, 1)
        return root

    def predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']


if __name__ == '__main__':
    dataset = pd.read_csv('datasetes/data_banknote_authentication.csv')
    decision_tree = DecisionTree(dataset, number_of_folds=5, max_depth=5, min_size=10,
                                 cost_function="ENTROPY")
    scores = decision_tree.run()
    print(scores)
    print(np.mean(scores))
