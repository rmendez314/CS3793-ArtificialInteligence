import numpy as np
import pandas as pd
import copy

# Data file name variables
id3_train = "data/id3-train.dat"
id3_weather_train = "data/id3-weather-train.dat"
id3_test = "data/id3-test.dat"

# read data from file
train_data = pd.read_csv(id3_train, delim_whitespace=True)
train_weather_data = pd.read_csv(id3_weather_train, delim_whitespace=True)
test_data = pd.read_csv(id3_test, delim_whitespace=True)


# Tree Node
class TreeNode:
    def __init__(self, parent=None):
        # which attr we need to split on, aka what we decided is the maxIG last node
        self.attr = None
        # children of this node, which are all the values of the feature
        self.children = []
        # which class value the feature corresponds to, either pure or majority
        self.label = None
        # value of parent's attribute
        self.splitVal = None
        # the number of instances of the current feature with the split val specified by the parent
        self.count = None
        # the parent node
        self.parent = None


# Pseudocode for the ID3 algorithm. Use this to create function(s).
def id3(data, root, attributes_remaining):
    # If you reach a leaf node in the decision tree and have no examples left or the examples are equally split among
    #   multiple classes
    if len(data) == 0:
        # Choose and the class that is most frequent in the entire training set and return the updated tree
        root.label = get_majority_label(data)
        return
    # If all the instances have only one class label aka pure class
    labels = data['class'].unique()
    if len(labels) == 1:
        # Make this as the leaf node and use the label as the class value of the node and return the updated tree
        root.label = labels[0]
        return
    # If you reached a leaf node but still have examples that belong to different classes (there are no remaining
    # attributes to be split)
    if len(attributes_remaining) == 0:
        # print("Majority",root.attr)
        # Assign the most frequent class among the instances at the leaf node and return the updated tree
        root.label = get_majority_label(data)
        return
    # Find the best attribute to split by calculating the maximum information gain from the attributes remaining by
    # calculating the entropy
    label = data.columns[-1]
    classes = labels
    max_attr = find_max_gain(data, label, classes, attributes_remaining)
    attributes_remaining.remove(max_attr)
    # Split the tree using the best attribute and recursively call the ID3 function using DFS to fill the sub-tree
    max_attr_values = data[max_attr].unique().tolist()
    max_attr_values.sort()

    for value in max_attr_values:
        subtable = data[data[max_attr] == value]
        child = TreeNode(parent=root)
        child.attr = max_attr
        child.splitVal = value
        child.count = len(subtable)
        root.children.append(child)
        attributes_copy = copy.deepcopy(attributes_remaining)
        id3(subtable, child, attributes_copy)
        # return the root as the tree
    return


# find total entropy for the data set
def find_entropy(data, label, classes):
    num_instances = data.shape[0]
    entropy = 0

    for c in classes:
        count = data[data[label] == c].shape[0]
        class_entropy = - (count / num_instances) * np.log2(count / num_instances)
        entropy += class_entropy
    return entropy


# find the entropy for each attribute
def find_entropy_attribute(data, label, classes):
    count = data.shape[0]
    entropy = 0

    for c in classes:
        label_class_count = data[data[label] == c].shape[0]
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count / count
            entropy_class = - probability_class * np.log2(probability_class)
        entropy += entropy_class
    return entropy


#  finds the info gain for a specific attribute
def get_info_gain(feature_name, data, label, classes):
    features = data[feature_name].unique()
    num_instances = data.shape[0]
    feature_info = 0
    # loop through each feature and calculate its info gain
    for feature in features:
        feature_data = data[data[feature_name] == feature]
        num_feature_instances = feature_data.shape[0]
        curr_entropy = find_entropy_attribute(feature_data, label, classes)
        feature_prob = num_feature_instances / num_instances
        feature_info += feature_prob * curr_entropy
    # return the value returned by find_entropy - feature_info
    return find_entropy(data, label, classes) - feature_info


# find max gain by calling and comparing each attributes info gain
def find_max_gain(data, label, classes, features):
    max_val = -1
    max_feature = None
    for feature in features:
        curr_feature_ig = get_info_gain(feature, data, label, classes)
        if max_val < curr_feature_ig:
            max_val = curr_feature_ig
            max_feature = feature
    return max_feature


# get the majority feature label
def get_majority_label(data):
    # get the class labels
    labels = data['class'].unique()
    # initialize the majority label to None
    majority_label = labels[0]
    maj_count = 0
    # go through each class label
    for label in labels:
        # get the count of the class label
        count = len(data[data['class'] == label])
        # if the count is greater than the majority label update the majority label
        if count > maj_count:
            majority_label = label
            maj_count = count
    # return the majority label
    return majority_label


# prediction for root instamce in data set
def predict_instance(root, data):
    node = None
    if root.label is not None:
        return root.label
    split = data[root.attr]
    for child in root.children:
        if child.splitVal == split:
            node = child
            break
    # if node is empty
    if node is None:
        node = root.children[0]
    return predict_instance(node, data)


# finds the accuracy for each data set
def get_accuracy(root, data):
    labels = data['class'].unique()
    correct = 0
    for i, row in data.iterrows():
        label = row['class']
        result = predict_instance(root.children[0], row)
        result2 = predict_instance(root.children[1], row)
        if label in [result, result2]:
            correct += 1
    accuracy = correct / len(data)
    return accuracy


# prints the decision tree
def print_tree(root, level=0):
    tab = '\t' * level
    if root.attr is None:
        for child in root.children:
            print_tree(child, level + 1)
    elif not root.children:
        print(tab, root.attr, "=", root.splitVal, ":", root.label, "--", root.count)
    else:
        print(tab, root.attr, "=", root.splitVal, ":")
        for child in root.children:
            print_tree(child, level + 1)


