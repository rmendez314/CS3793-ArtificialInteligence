import main
import tensorflow as tf
import keras


def rnn_load_data():
    # Load the Reuters dataset - use the Keras version
    # Select the vocabulary size while loading the data
    # The data will be loaded as integer representations for each word

    # Load the data - training as well as testing
    (x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data()

    # Prepare the data to be used for the next steps
    # Each data entry (newswire) can be of different lengths
    # Make each newswire consistent - same number of words
    # Pad' words (say 0) to get to the standard length or remove words

    words = tf.keras.datasets.reuters.get_word_index(path='reuters_word_index.json')

    # Get the vocabulary size
    vocab_size = len(words) + 1
    # Get the length of the longest newswire
    max_length = max([len(x) for x in x_train])
    # Pad the newswires to the same length
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length, padding='post')
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length, padding='post')

    # # Convert the labels to categorical
    # y_train = keras.utils.to_categorical(y_train, num_classes=46)
    # y_test = keras.utils.to_categorical(y_test, num_classes=46)
    return x_train, x_test, y_train, y_test, vocab_size, max_length
#
# def rnn_visualization():
#
#
# def rnn_test_model():
#
#
# def rnn_train_model():


# function to create the model
def rnn_create_model():
    # Create the model
    # Use the Keras functional API
    # Input layer
    # Embedding layer
    # LSTM layer
    # Output layer
    # Use the softmax activation function
    # Use the categorical crossentropy loss function
    # Use the Adam optimizer
    # Use the accuracy metric
    # Use the model summary method
    # Use the model summary method to print the model summary
    vocab_size = 5000
    embedding_dim = 128
    max_length = 500
    hidden_units = 128
    num_classes = 46

    # Create the model
    model = keras.models.Sequential()

    # Input layer
    model.add(keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))

    # LSTM layer
    model.add(keras.layers.LSTM(units=hidden_units, activation='tanh', recurrent_activation='hard_sigmoid',
                                use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                                bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
                                recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                                dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False,
                                return_state=False, go_backwards=False, stateful=False, unroll=False))

    # Output layer
    model.add(keras.layers.Dense(units=num_classes, activation='softmax'))

    # Use the softmax activation function
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
