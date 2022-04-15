# import all required libraries
import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from pylab import rcParams
# %matplotlib inline
import computer_vision
import RNN_NLP


# main function
def main():
    # computer_vision.cnn_load_data()
    RNN_NLP.rnn_load_data()
    # model = RNN_NLP.rnn_create_model()
    # print(model.summary())

if __name__ == "__main__":
    main()