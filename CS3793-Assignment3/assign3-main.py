import id3_decision_tree
import gradient_descent
import pprint as pp


def gradient_descent_main():
    # Main algorithm loop
    # Loop through all the different learning rates [0.05, 1]
    learning_rates = [.05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95, 1]
    for lr in learning_rates:
        # For each learning rate selected, call the gradient descent function to
        #   obtain the train and test accuracy values
        results = gradient_descent.gradient_descent(gradient_descent.train_data, gradient_descent.test_data, lr)
        print("Test with learning rate:", lr)
        # Print both the accuracy values as "Accuracy for LR of 0.1 on Training
        #   set = x %" OR "Accuracy for LR of 0.1 on Testing set = x %"
        print(f"train_accuracy: {results[0]}%")
        print(f"test_accuracy: {results[1]}%")
        print()


def id3_main():
    # Main algorithm loop
    train_attributes = list(id3_decision_tree.train_data)[0:-1]
    test_attributes = list(id3_decision_tree.test_data)[0:-1]

    # ID3 algorithm for the training data
    print("Decision Tree: Training Data")
    train_root = id3_decision_tree.TreeNode()
    id3_decision_tree.id3(id3_decision_tree.train_data, train_root, train_attributes)
    id3_decision_tree.print_tree(train_root)

    print()

    # Id3 algorithm for the test data
    print("Decision Tree: Test Data")
    test_root = id3_decision_tree.TreeNode()
    id3_decision_tree.id3(id3_decision_tree.train_data, test_root, test_attributes)
    id3_decision_tree.print_tree(test_root)

    # gets the accuracy for the training and test data
    train_accuracy = id3_decision_tree.get_accuracy(train_root, id3_decision_tree.train_data)
    test_accuracy = id3_decision_tree.get_accuracy(test_root, id3_decision_tree.test_data)
    print()
    print("Accuracy on the Train data == ", train_accuracy)
    print("Accuracy on the Test data == ", test_accuracy)


if __name__ == '__main__':
    gradient_descent_main()
    id3_main()
