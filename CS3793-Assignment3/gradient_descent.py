import pandas as pd
from scipy.special import expit
import copy
import numpy as np

# Data file name variables
train = "data/gd-train.dat"
test = "data/gd-test.dat"

# Read the training and testing data files
train_data = pd.read_csv(train,delim_whitespace=True)
test_data = pd.read_csv(test, delim_whitespace=True)

# Activation Function - implement Sigmoid


def activation_function(h):
    # given 'h' compute and return 'z' based on the activation function implemented
    return expit(h)

# Train the model using the given training dataset and the learning rate
# return the "weights" learnt for the perceptron - include the weight assocaited with bias as the last entry


def train(train_data, learning_rate=0.05):
  # initialize weights to 0
  w = np.zeros(train_data.iloc[0].shape)
  # go through each training data instance
  for i, row in train_data.iterrows():
    # get 'x' as one multi-variate data instance and 'y' as the ground truth class label
    x = copy.deepcopy(train_data.iloc[i][0:13])
    x['bias'] = 1
    y = copy.deepcopy(train_data.iloc[i][-1])
    # obtain h(x)
    h = np.dot(w, x)
    # call the activation function with 'h' as parameter to obtain 'z'
    z = activation_function(h)
    # update all weights individually using learning_rate, (y-z), and the corresponding 'x'
    for j in range(len(w)):
      w[j] = w[j] + learning_rate * (y - z) * x[j]
  # return the final learnt weights
  return w

# Test the model (weights learnt) using the given test dataset
# return the accuracy value


def test(test_data, weights, threshold):
    accuracy = 0
    # go through each testing data instance
    for i,row in test_data.iterrows():
        # row = np.array(test_data.iloc[i])
        # get 'x' as one multi-variate data instance and 'y' as the ground truth class label
        x = copy.deepcopy(test_data.iloc[i][0:13])
        x['bias'] = 1
        y = copy.deepcopy(test_data.iloc[i][13])
        # obtain h(x)
        h = np.dot(weights, x)
        # call the activation function with 'h' as parameter to obtain 'z'
        z = activation_function(h)
        # use 'threshold' to convert 'z' to either 0 or 1 so as to match to the ground truth binary labels
        if z < threshold:
            z = 0
        else:
            z = 1
        # compare the thresholded 'z' with 'y' to calculate the positive and negative instances for calculating accuracy
        if z == y:
            accuracy = accuracy + 1
    accuracy = (accuracy / len(test_data)) * 100
    # return the accuracy value for the given test dataset
    return accuracy

# Gradient Descent function


def gradient_descent(df_train, df_test, learning_rate=0.05, threshold=0.5):
    train_accuracy = 0
    test_accuracy = 0
    # call the train function to train the model and obtain the weights
    train_weights = train(df_train, learning_rate)
    test_weights = train(df_test, learning_rate)
    # call the test function with the training dataset to obtain the training accuracy
    train_accuracy = test(df_train, train_weights, threshold)
    # call the test function with the testing dataset to obtain the testing accuracy
    test_accuracy = test(df_test, test_weights, threshold)
    # return (trainAccuracy, testAccuracy)
    return (train_accuracy, test_accuracy)