import math

from DecisonTree import Leaf, Question, DecisionNode, class_counts
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list, min_for_pruning=0, target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()
        self.min_for_pruning = min_for_pruning

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        counts = class_counts(rows, labels)
        impurity = 0.0
        total = len(rows)
        for label in counts:
            p = counts[label] / total
            impurity -= p * math.log2(p)

        return impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = current_uncertainty
        # ====== YOUR CODE: ======
        total = len(right) + len(left)
        right_entropy = ID3.entropy(right, right_labels)
        left_entropy = ID3.entropy(left, left_labels)

        info_gain_value = - (right_entropy * len(right) / total) - (left_entropy * len(left) / total)
        # ========================

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """

        gain, true_rows, true_labels, false_rows, false_labels = None, None, None, None, None
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        true_rows, true_labels = [rows[idx_row] for idx_row in range(len(rows)) if question.match(rows[idx_row])], [
            labels[idx_row] for idx_row in range(len(rows)) if question.match(rows[idx_row])]
        false_rows, false_labels = [rows[idx_row] for idx_row in range(len(rows)) if
                                    not question.match(rows[idx_row])], [labels[idx_row] for idx_row in range(len(rows))
                                                                         if not question.match(rows[idx_row])]
        gain = self.info_gain(true_rows, true_labels, false_rows, false_labels, current_uncertainty)

        return gain, true_rows, true_labels, false_rows, false_labels

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        for feature in range(rows.shape[1]):
            values = np.sort(rows[:, feature])
            for label in set(labels):
                for value in range(len(values) - 1):
                    average = (values[value] + values[value + 1]) * 0.5
                    question = Question(label, feature, average)
                    gain, true_rows, true_labels, false_rows, false_labels = \
                    self.partition(rows, labels, question, current_uncertainty)
                    if gain >= best_gain:
                        best_gain = gain
                        best_question = question
                        best_false_rows, best_false_labels = false_rows, false_labels
                        best_true_rows, best_true_labels = true_rows, true_labels

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        best_question = None
        true_branch, false_branch = None, None

        if len(labels) <= 1:
            return Leaf(rows, labels)
        best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = self.find_best_split(rows, labels)

        if best_gain == 0 or len(best_true_rows) < self.min_for_pruning or len(best_false_rows) < self.min_for_pruning:
            return Leaf(rows, labels)

        true_branch = self.build_tree(np.array(best_true_rows), np.array(best_true_labels))
        false_branch = self.build_tree(np.array(best_false_rows), np.array(best_false_labels))

        return DecisionNode(best_question, true_branch, false_branch)

    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        self.tree_root = self.build_tree(x_train, y_train)


    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        if node is None:
            node = self.tree_root
        prediction = None

        # ====== YOUR CODE: ======
        if isinstance(node, Leaf):
            return max(node.predictions, key=node.predictions.get)

        if node.question.match(row):
            return self.predict_sample(row, node.true_branch)
        else:
            return self.predict_sample(row, node.false_branch)        # ========================

        return prediction

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        y_pred = None

        # ====== YOUR CODE: ======
        y_pred = np.array([self.predict_sample(row) for row in rows])
        # ========================

        return y_pred
