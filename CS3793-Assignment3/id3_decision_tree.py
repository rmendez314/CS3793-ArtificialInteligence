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


class Node:
    def __init__(self, parent=None):
        self.parent = None
        self.attribute = None
        self.label = None
        self.attr_value = None
        self.count = None
        self.children = []


class DecisionTreeClassifier:
    """Decision Tree Classifier using ID3 algorithm."""

    def __init__(self, X, feature_names, labels):
        self.X = X  # features or predictors
        self.feature_names = feature_names  # name of the features
        self.labels = labels  # categories
        self.labelCategories = list(set(labels))  # unique categories
        # number of instances of each category
        self.labelCategoriesCount = [list(labels).count(x) for x in self.labelCategories]
        self.node = None  # nodes
        # calculate the initial entropy of the system
        self.entropy = ([x for x in range(len(self.labels))])


# function to build a decision tree using id3 algorithm
def id3(data, attributes, target_attribute_name, parent_examples=None):
    # If all examples are positive, Return the single-node tree Root, with label = +.
    if data['class'].value_counts().min() == len(data):
        return Node()
    # To make the code generic, changing target variable class name
    Class = data.keys()[-1]
    print(parent_examples)
    # If the dataset is empty or the attributes list is empty, return the majority class.
    if len(data) == 0 or len(attributes) == 0:
        return get_majority_label(data)
    # If all the records in the dataset belongs to same class, return that class.
    elif data[Class].nunique() == 1:
        return data[Class].iloc[0]
    # If the dataset has no more attributes, return the majority class.
    elif len(attributes) == 0:
        return get_majority_label(data)
    # If none of the above conditions is true, grow the tree.
    else:
        # Get the attribute that maximizes the information gain.
        attribute = find_max_gain(data, attributes)
        # Get the possible values of the attribute.
        values = data[attribute].unique()
        # Create a dictionary to store the subtree built using id3() method for corresponding attribute-value pair.
        tree = {attribute: {}}
        # Iterate over all the values of the attribute.
        for value in values:
            # Get the examples where the attribute has the corresponding value.
            examples = get_subtable(data, attribute, value)
            # If the examples empty, then assign the majority class label to the current node.
            if len(examples) == 0:
                tree[attribute][value] = get_majority_label(data)
            # Else grow a subtree for the current value and add it to the dictionary.
            else:
                remaining_attributes = [i for i in attributes if i != attribute]
                tree[attribute][value] = id3(examples, remaining_attributes, target_attribute_name, attribute)
            return tree


# function to find attribute that maximizes the info gain
def find_max_gain(data, attribute_list):
    # To make the code generic, changing target variable class name
    Class = data.keys()[-1]
    # Initialize the information gain and attribute
    info_gain = 0
    attr = None
    # Calculate the information gain for each attribute and
    # return the attribute with maximum information gain
    for attribute in attribute_list:
        info_gain_temp = find_entropy_attributes(data, attribute) - find_entropy(data)
        if info_gain_temp > info_gain:
            info_gain = info_gain_temp
            attr = attribute
    return attr


def find_winning_attr(data):
    entropy_attr = []
    ig = []
    for attr in data.keys()[:-1]:
        ig.append(find_entropy(data) - find_entropy_attributes(data, attr))
    return data.keys()[:-1][np.argmax(ig)]


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


def get_subtable(data, node, value):
    return data[data[node] == value].reset_index(drop=True)


# def build_tree(data, tree=None):
#     # To make the code generic, changing target variable class name  #Here we build our
#     #   decision tree  #Get attribute with maximum information gain
#     decisions = data.keys()[-1]
#     # Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
#     node = find_winning_attr(data)
#     # Create an empty dictionary to create tree
#     attr_values = np.unique(data[node])
#     if tree is None:
#         tree = {node: {}}
#         for value in attr_values:
#             subtable = get_subtable(data, node, value)
#             dec_value, counts = np.unique(subtable['class'], return_counts=True)
#             if len(counts) == 1:  # Checking purity of subset
#                 tree[node][value] = dec_value[0]
#             else:
#                 tree[node][value] = build_tree(subtable)  # Calling the function recursively
#     return tree


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
