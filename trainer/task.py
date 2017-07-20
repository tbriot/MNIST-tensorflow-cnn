import argparse
import pandas as pd
import numpy as np
import trainer.model as m
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO following parameters should be passed as args from the CLI :
# - data dir
# - chkp dir
# - event dir

ROOTDIR = "C:/Users/Timo/PycharmProjects/Kaggle/MNIST-tensorflow-cnn/"
DATADIR = ROOTDIR + "data/"
CHECKPOINT_DIR = ROOTDIR + "tf-model-checkpoint/"
EVENTS_DIR = ROOTDIR + "tf-event-files/"

TRAIN_FILE = "train.csv"

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


def run(data_dir, event_dir, chkp_dir, load_last_chkp):

    # load data set
    data, labels = load_data_from_file(data_dir + TRAIN_FILE, rows=50000)

    # define our CNN model layout
    layers_layout = [{"type": "conv", "filter_size": 5, "depth": 6,  "mp_size": 2},
                     {"type": "conv", "filter_size": 5, "depth": 16, "mp_size": 2},
                     {"type": "drop"},
                     {"type": "full", "units": 120, "activation": True},
                     {"type": "full", "units": 84, "activation": True},
                     {"type": "full", "units": 10, "activation": False}]

    # build the model
    model = m.CnnModel(layers_layout)

    """
    training_program = [{"lr": 1E-3, "epochs": 3},
                        {"lr": 5E-4, "epochs": 4},
                        {"lr": 1E-4, "epochs": 5},
                        {"lr": 5E-5, "epochs": 6}]
    """

    training_program = [{"lr": 1E-3, "epochs": 2}]

    model.train(data, labels,
                training_program=training_program,
                event_file_dir=event_dir,
                chk_file_dir=chkp_dir,
                load_last_chkp=load_last_chkp,
                batch=100,
                val_set_size=0.1,
                keep_prob=0.5,
                report_freq=100,
                debug=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",
                        required=True,
                        type=str,
                        help="Training files directory local or GCS",
                        nargs="?")
    parser.add_argument("--event-dir",
                        required=True,
                        type=str,
                        help="TF event files directory local or GCS",
                        nargs="?")
    parser.add_argument("--chkp-dir",
                        required=True,
                        type=str,
                        help="TF checkpoint files directory local or GCS",
                        nargs="?")
    parser.add_argument("--load_last_chkp",
                        required=False,
                        type=str,
                        help="Whether or not the last training checkpoint should be loaded",
                        nargs="?",
                        default="False")
    args, _ = parser.parse_known_args()
    args_dic = args.__dict__

    run(args_dic["data_dir"],
        args_dic["event_dir"],
        args_dic["chkp_dir"],
        True if args_dic["load_last_chkp"] == "True" else False)
