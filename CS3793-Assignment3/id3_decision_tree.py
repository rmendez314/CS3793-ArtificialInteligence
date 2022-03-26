
import numpy as np
import pandas as pd

eps = np.finfo(float).eps

# Data file name variables
id3_train = "data/id3-train.dat"
id3_weather_train = "data/id3-weather-train.dat"
id3_test = "data/id3-test.dat"

train_data = pd.read_csv(id3_train, delim_whitespace=True)
train_weather_data = pd.read_csv(id3_weather_train, delim_whitespace=True)
test_data = pd.read_csv(id3_test, delim_whitespace=True)


# function to build the decision tree recursively using the ID3 algorithm
def ID3(data, features):
    # get the class labels
    labels = data['class'].unique()
    # if the data is empty or the features is empty
    # return the majority class label
    if len(data) == 0 or len(features) == 0:
        return get_majority_label(data)
    # if the data is homogeneous
    # return the class label
    if len(labels) == 1:
        return labels[0]
    # get the best feature
    best_feature = find_winning_attr(data)
    # why do i have to add this, what the heck :(
    if best_feature is None:
        if len(features) == 1:
            best_feature = features[0]
    # initialize the tree
    tree = {best_feature: {}}
    # get the unique values of the best feature
    values = data[best_feature].unique()
    # go through each value of the best feature
    for value in values:
        # get the data subset for the current value of the best feature
        subset = data[data[best_feature] == value]
        # get the best feature subset
        best_feature_subset = features[:]
        # remove the best feature from the best feature subset
        best_feature_subset.remove(best_feature)
        # build the decision tree for the best feature subset
        tree[best_feature][value] = ID3(subset, best_feature_subset)
    # return the tree
    return tree
# calculates entropy for data


# get the majority class label
def get_majority_label(data):
    # get the class labels
    labels = data['class'].unique()
    # initialize the majority label to None
    majority_label = None
    # go through each class label
    for label in labels:
        # get the count of the class label
        count = data['class'].value_counts()[label]
        # if the count is greater than the majority label
        # update the majority label
        if majority_label is None or count > majority_label:
            majority_label = label
    # return the majority label
    return majority_label


def find_entropy(data):
    y_attr = data.keys()[-1]
    # print(y_attr)
    entropy = 0
    values = data[y_attr].unique()
    for value in values:
        fraction = data[y_attr].value_counts()[value] / len(data[y_attr])
        entropy += -fraction * np.log2(fraction)
    return entropy


def find_entropy_attributes(df, attribute):
    # To make the code generic, changing target variable class name
    Class = df.keys()[-1]
    # This gives all '1' and '0'
    target_variables = df[Class].unique()
    # This gives different features in that attribute (like 'Hot','Cold' in Temperature
    variables = df[attribute].unique()
    entropy2 = 0
    den = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            fraction = num / (den + eps)
            entropy += -fraction * np.log2(fraction + eps)
        fraction2 = den / len(df)
        entropy2 += -fraction2 * entropy
    return abs(entropy2)


def find_winning_attr(data):
    entropy_attr = []
    ig = []
    for attr in data.keys()[:-1]:
        ig.append(find_entropy(data) - find_entropy_attributes(data, attr))
    return data.keys()[:-1][np.argmax(ig)]


def get_subtable(data, node, value):
    return data[data[node] == value].reset_index(drop=True)


# def build_tree(data, attributes):
#     labels = data.keys()[-1].unique()
#     if len(data) == 0 or len(attributes) == 0:
#         return 0
#     if len(labels) == 1:
#         return labels[0]
#     best_attr = find_winning_attr(data)
#     tree = {best_attr: {}}
#     print(data[best_attr].head())
#     values = data[best_attr].unique()
#     for value in values:
#         subtable = get_subtable(data, best_attr, value)
#         best_attr_subset = attributes[:]
#         best_attr_subset.remove(best_attr)
#         tree[best_attr][value] = build_tree(subtable, best_attr_subset)
#
#     return tree

def build_tree(data, tree=None):
    # To make the code generic, changing target variable class name  #Here we build our
    #   decision tree  #Get attribute with maximum information gain
    decisions = data.keys()[-1]
    # Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    node = find_winning_attr(data)
    # Create an empty dictionary to create tree
    attr_values = np.unique(data[node])
    if tree is None:
        tree = {node: {}}
        for value in attr_values:
            subtable = get_subtable(data, node, value)
            dec_value, counts = np.unique(subtable['class'], return_counts=True)
            if len(counts) == 1:  # Checking purity of subset
                tree[node][value] = dec_value[0]
            else:
                tree[node][value] = build_tree(subtable)  # Calling the function recursively
    return tree


# FUNCTION TO PRINT DICTIONARY recursively
def print_tree(tree, level=0):
    # get the keys of the tree
    keys = list(tree.keys())
    # get the values of the tree
    values = list(tree.values())
    # get the best feature
    best_feature = keys[0]
    # get the values of the best feature
    best_feature_values = list(tree[best_feature].keys())
    # get the values of the best feature
    best_feature_subtrees = list(tree[best_feature].values())
    # print the best feature
    print("{}{} =".format("\t" * level, best_feature))

    # go through each value of the best feature
    for i in range(len(best_feature_values)):
        # print the value of the best feature
        print("{}{} = {}".format("\t" * (level + 1),
                                 best_feature_values[i], best_feature_subtrees[i]))
        # if the value of the best feature is a dictionary
        if isinstance(best_feature_subtrees[i], dict):
            # print the dictionary recursively
            print_tree(best_feature_subtrees[i], level + 2)
# function to print the decision tree


# def print_tree(root, level=0):
#     # if the node is a leaf node
#     if root.label is not None:
#         print('\t' * level, 'Leaf Node:', root.label)
#     else:
#         print('\t' * level, 'Node:', root.attribute, '=', root.attr_value)
#         for child in root.children:
#             print_tree(child, level + 1)

# calculate information gain for a specific feature
def get_info_gain(data, feature):
    # get the unique values of the feature
    values = data[feature].unique()
    # initialize the info gain to 0
    info_gain = 0
    # go through each value of the feature
    for value in values:
        # get the data subset for the current value of the feature
        subset = data[data[feature] == value]
        # calculate the entropy of the subset
        subset_entropy = find_entropy(subset)
        # calculate the info gain using the formula
        info_gain += (len(subset) / len(data)) * subset_entropy
    # return the info gain value
    return find_entropy(data) - info_gain
