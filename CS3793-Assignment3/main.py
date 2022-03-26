import id3_decision_tree
import gradient_descent


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
    num = 2
    print("Using Train Data:")
    print("---------------------------------------")
    entropy = id3_decision_tree.find_entropy(id3_decision_tree.train_data)
    print(f"entropy: {entropy}")

    # attr_entropy = id3_decision_tree.find_entropy_attributes(id3_decision_tree.train_data,
    #                                                          id3_decision_tree.train_data.keys()[num])
    num_attr = len(id3_decision_tree.train_data.keys()) - 1
    keys = id3_decision_tree.train_data.keys()
    for attr in keys:
        print(f"{attr}: {id3_decision_tree.find_entropy_attributes(id3_decision_tree.train_data, attr)}")
        print(f"info gain for {attr}: {id3_decision_tree.get_info_gain(id3_decision_tree.train_data, attr)}")
        print()

    majority_label = id3_decision_tree.get_majority_label(id3_decision_tree.train_data)
    print(f"majority label: {majority_label}")
    # print(f"attr{num} = {attr_entropy}")

    winner = id3_decision_tree.find_winning_attr(id3_decision_tree.train_data)
    print(f"winning attr: {winner}")

    print("Train Data:\n")
    # features = id3_decision_tree.train_data.keys()[:num_attr]
    # tree = id3_decision_tree.ID3(id3_decision_tree.train_data, features)
    tree1 = id3_decision_tree.build_tree(id3_decision_tree.train_data.head(20))
    id3_decision_tree.print_tree(tree1)

    print()

    print("Using Train Weather Data:")
    print("---------------------------------------")
    entropy = id3_decision_tree.find_entropy(id3_decision_tree.train_weather_data)
    print(f"entropy: {entropy}")

    keys = id3_decision_tree.train_weather_data.keys()
    for attr in keys:
        print(f"{attr}: {id3_decision_tree.find_entropy_attributes(id3_decision_tree.train_weather_data, attr)}")
        print(f"info gain for {attr}: {id3_decision_tree.get_info_gain(id3_decision_tree.train_weather_data, attr)}")
        print()

    majority_label = id3_decision_tree.get_majority_label(id3_decision_tree.train_weather_data)
    print(f"majority label: {majority_label}")

    winner = id3_decision_tree.find_winning_attr(id3_decision_tree.train_weather_data)
    print(f"winning attr: {winner}")

    print("Test Data:\n")
    tree2 = id3_decision_tree.build_tree(id3_decision_tree.train_weather_data)
    id3_decision_tree.print_tree(tree2)


if __name__ == '__main__':
    # gradient_descent_main()
    id3_main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
