from matplotlib import pyplot

import main
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.losses import sparse_categorical_crossentropy

from tensorflow.keras.optimizers import Adam

checkpoint_path = "/cifar"



def cnn_load_data():
    # Load the data - training as well as testing
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode='fine')
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    # Prepare the data that can be used by the next step - creating and training the DL model
    # The data from TensforFlow and Keras will only have integer class labels. Each of those 100 integer class
    # labels correspond to the following names, in the correct order
    fine_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                   'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                   'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                   'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                   'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                   'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
                   'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
                   'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
                   'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
                   'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
                   'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    # These are the string labels for the 20 superclasses. You may not need to use this at all, just provided here for
    # reference.
    coarse_labels = ['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
                     'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
                     'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
                     'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals', 'trees',
                     'vehicles_1', 'vehicles_2']

    cnn_visualization(x_train, y_train, x_test, y_test, coarse_labels, fine_labels)
    cnn_create_model(x_train, y_train, x_test, y_test)


def cnn_visualization(x_train, y_train, x_test, y_test, course_labels, fine_labels):
    # Visualize the data by plotting 100 random images, one each for the 100 classes
    # Draw 10 images in one row, 10 rows total
    print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

    # plot 100 random images from the training set
    for i in range(100):
        # pick a random image
        idx = np.random.randint(0, len(x_train))
        # plot the image
        pyplot.subplot(10, 10, i + 1)
        pyplot.axis('off')
        pyplot.imshow(x_train[idx])
        # # plot the label
        pyplot.title(fine_labels[y_train[idx][0]])

    pyplot.show()


def cnn_test_model(x_train, y_train, x_test, y_test):
    # Re-initialize the model
    test_model = cnn_create_model()
    test_model.load_weights(checkpoint_path)
    # Evaluate the trained DL model on the CIFAR-100 test dataset
    # Parse numbers as floats
    input_train = x_train.astype('float32')
    input_test = x_test.astype('float32')

    # Normalize data
    input_train = input_train / 255
    input_test = input_test / 255
    loss_function = sparse_categorical_crossentropy
    optimizer = Adam()

    # Compile the model
    test_model.compile(loss=loss_function,
                       optimizer=optimizer,
                       metrics=['accuracy'])

    score = test_model.evaluate(input_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


# Train/fit the DL model using the training CIFAR-100 data
def cnn_train_model(model, x_train, y_train, x_test, y_test, checkpoint_path):
    # Model configuration
    batch_size = 50
    loss_function = sparse_categorical_crossentropy
    no_epochs = 100
    optimizer = Adam()
    validation_split = 0.2
    verbosity = 1
    # Parse numbers as floats
    input_train = x_train.astype('float32')
    input_test = x_test.astype('float32')

    # Normalize data
    input_train = input_train / 255
    input_test = input_test / 255
    # Compile the model
    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # save weights to checkpoint
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Fit data to model
    history = model.fit(input_train, y_train,
                        batch_size=batch_size,
                        epochs=no_epochs,
                        verbose=verbosity,
                        callbacks=model_checkpoint_callback,
                        validation_split=validation_split)

    cnn_plot_accuracy(history)


# Plot the training/validation accuracy and loss
def cnn_plot_accuracy(history):
    # Plot history: Accuracy
    plt.plot(history.history['val_accuracy'])
    plt.title('Validation accuracy history')
    plt.ylabel('Accuracy value (%)')
    plt.xlabel('No. epoch')
    plt.show()


# Create a DL model for Computer Vision - Convolutional Neural Network (Use *TensorFlow* and *keras*, as shown in the example code in the lecture for 'deep-learning')
def cnn_create_model():
    img_width, img_height, img_num_channels = 32, 32, 3
    no_classes = 100
    # Determine shape of the data
    input_shape = (img_width, img_height, img_num_channels)
    # Create the model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))

    return model

