import pandas as pd
import numpy as np
from trainer import model as m
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ROOTDIR = "C:/Users/Timo/PycharmProjects/Kaggle/MNIST-tensorflow-cnn/"
DATADIR = ROOTDIR + "data/"
TRAIN_SET = DATADIR + "train.csv"
EVENTS_DIR = ROOTDIR + "tf-event-files/"

IMAGE_H, IMAGE_W, IMAGE_C = 28, 28, 1  # height, width, channels (grayscale = just one channel)

# TODO compile tf library on machine
# Warning being :
# The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could
# speed up CPU computations.


# normalize data so that mean=0 and std dev=1
# data is a numpy ndarray
def normalize_data(data):
    data = data - data.mean()
    return data / data.std()


# the training set contains 42k images
def load_data_from_file(filename, rows=2000):
    data = pd.read_csv(filename, nrows=rows)  # load data in a pandas Dataframe
    labels = data.pop("label")  # extract labels from the dataframe
    labels = pd.get_dummies(labels)  # convert labels into one-hot vectors

    # convert pandas Dataframes into float32 numpy arrays
    data = data.values.astype(np.float32)
    labels = labels.values.astype(np.float32)

    data = normalize_data(data)  # set data mean=0 and std dev=1
    data = data.reshape(-1, IMAGE_H, IMAGE_W, IMAGE_C)  # reshape to NHWC format

    return data, labels


def main():

    # load data set
    data, labels = load_data_from_file(TRAIN_SET, rows=20000)

    # define our CNN model layout
    layers_layout = [{"type": "conv", "filter_size": 5, "depth": 6,  "mp_size": 2},
                     {"type": "conv", "filter_size": 5, "depth": 12, "mp_size": 2},
                     {"type": "drop"},
                     {"type": "full", "units": 48, "activation": True},
                     {"type": "full", "units": 24, "activation": True},
                     {"type": "full", "units": 10, "activation": False}]

    # build the model
    model = m.CnnModel(layers_layout)

    training_program = [{"lr": 1E-3, "epochs": 1},
                        {"lr": 5E-4, "epochs": 1}]
    model.train(data, labels,
                training_program=training_program,
                event_file_dir=EVENTS_DIR,
                batch=100,
                val_set_size=0.1,
                keep_prob=0.5,
                report_freq=100)

if __name__ == "__main__":
    main()
