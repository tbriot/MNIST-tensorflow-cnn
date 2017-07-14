# what is a tf meta graph ?
# TODO use GPU
# TODO normalize with tf.nn.batch_normalization

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time


ROOTDIR = "C:/Users/Timo/PycharmProjects/Kaggle/MNIST-tensorflow-cnn/"
DATADIR = ROOTDIR + "data/"
TF_EVTS_DIR = ROOTDIR + "tf-event-files/"
CHECKPOINT_DIR = ROOTDIR + "tf-model-checkpoint/"
TRAIN_SET = DATADIR + "train.csv"
TEST_SET = DATADIR + "test.csv"

HEIGHT = 28  # image height
WIDTH = HEIGHT
CHANNELS = 1  # grayscale image has just one channel

LABELS = 10  # number of labels

DROPOUT = 0.5


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
    data = data.reshape(-1, HEIGHT, WIDTH, CHANNELS)

    return data, labels


# the testing set contains 28k images
def load_test_data_from_file(filename):
    data = pd.read_csv(filename)  # load data in a pandas Dataframe

    # convert pandas Dataframes into float32 numpy arrays
    data = data.values.astype(np.float32)

    data = normalize_data(data)  # set data mean=0 and std dev=1
    data = data.reshape(-1, HEIGHT, WIDTH, CHANNELS)

    return data


# convolutional layer
def conv_layer(input_data, in_channels, out_channels, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, in_channels, out_channels], stddev=0.1), name="W")  # filters are 5x5
        b = tf.Variable(tf.constant(0.1, shape=[out_channels]), name="b")
        conv = tf.nn.conv2d(input_data, w, strides=[1, 1, 1, 1], padding="SAME", name="conv")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w, collections=["train", "predict"])
        tf.summary.histogram("biases", b, collections=["train"])
        tf.summary.histogram("activation", act, collections=["train"])
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # max pooling 1:2


# fully connected layer
def fc_layer(input_data, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[size_out]))
        act = tf.matmul(input_data, w) + b
        tf.summary.histogram("weights", w, collections=["train"])
        tf.summary.histogram("biases", b, collections=["train"])
        tf.summary.histogram("activation", act, collections=["train"])
        return act


# model
def build_model(learning_rate, size_conv1, size_conv2, size_fc1, size_fc2):
    # setup placeholders
    x = tf.placeholder(tf.float32, shape=[28000, HEIGHT, WIDTH, CHANNELS], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="y")

    # dropout
    keep_prob = tf.placeholder('float', name="keep_prob")

    conv1 = conv_layer(x, 1, size_conv1, "conv1")
    conv_out = conv_layer(conv1, size_conv1, size_conv2, "conv2")
    conv_out = tf.nn.dropout(conv_out, keep_prob)

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * size_conv2])

    fc1 = fc_layer(flattened, 7 * 7 * size_conv2, size_fc1, "fc1")
    relu1 = tf.nn.relu(fc1)

    fc2 = fc_layer(relu1, size_fc1, size_fc2, "fc2")
    relu2 = tf.nn.relu(fc2)

    logits = fc_layer(relu2, size_fc2, LABELS, "fc3")

    with tf.name_scope("predict"):
        tf.argmax(logits, axis=1, name="pred")

    with tf.name_scope("xent"):
        xent = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits, labels=y),
                    name="xent"
        )
        tf.summary.scalar("xent", xent, collections=["train"])
        # track xent on x-validation set in a different summary scalar
        tf.summary.scalar("xent_val", xent, collections=["xval"])

    with tf.name_scope("train"):
        tf.train.AdamOptimizer(learning_rate).minimize(xent, name="train_step")

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy, collections=["train"])
        # track accuracy on x-validation set in a different summary scalar
        tf.summary.scalar("accuracy_val", accuracy, collections=["xval"])

    return tf.get_default_graph()


def train_model(g, data, labels, steps=2000, batch=100, run_name="run", restore_checkpoint=False):

    # put aside a cross-validation data set, size = 10 pct
    data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=0.2)
    print("Training set size is %d. Cross-validation set size is %d" % (data_train.shape[0], data_val.shape[0]))

    with g.as_default():
        # merge summaries for training and x-validation purpose
        sum_train = tf.summary.merge_all("train")
        sum_xval = tf.summary.merge_all("xval")
        # instantiate tf summaries file writer
        writer = tf.summary.FileWriter(TF_EVTS_DIR + run_name, graph=g)

        # create a saver to persist the trained model on disk
        saver = tf.train.Saver()

        fd_val = {g.get_tensor_by_name("x:0"): data_val,
                  g.get_tensor_by_name("y:0"): labels_val,
                  g.get_tensor_by_name("keep_prob:0"): 1.0}  # no dropout when x-validating

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if restore_checkpoint:
            checkpoint_state = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
            saver.restore(sess, checkpoint_state.model_checkpoint_path)

        print("Training is starting")

        for step in range(1, steps+1):
            # get random batches of images and labels
            batch_data, _, batch_labels, _ = train_test_split(data, labels, train_size=batch)

            # create feed dictionary with batches
            fd = {g.get_tensor_by_name("x:0"): batch_data,
                  g.get_tensor_by_name("y:0"): batch_labels,
                  g.get_tensor_by_name("keep_prob:0"): DROPOUT}

            # every 10 steps get tf summarries and save them to disk (and still train the model)
            if step % 10 == 0:
                _, sum_tr = sess.run([g.get_operation_by_name("train/train_step"), sum_train],
                                     feed_dict=fd)
                writer.add_summary(sum_tr, step)
            else:
                # train the model only, don't compute summaries
                sess.run([g.get_operation_by_name("train/train_step")], feed_dict=fd)

            # every 100 steps compute accuracy on x-validation set
            if step % 100 == 0:
                sum_val = sess.run(sum_xval, feed_dict=fd_val)
                writer.add_summary(sum_val, step)
                print("Step %d completed" % step)

        # save model to disk once the training is complete
        saver.save(sess, CHECKPOINT_DIR + run_name)
        print("Trained model saved to disk")


# make a prediction based on a model saved on the disk
def make_prediction(model):
    # load test data in memory (28k records), normalize it and return a numpy array
    test_data = load_test_data_from_file(TEST_SET)

    with model.as_default() as g:

        # create a saver to restore the trained model tf Variables from disk
        saver = tf.train.Saver()

        # get filepath of the model checkpoint saved on disk
        model_filepath = tf.train.latest_checkpoint(CHECKPOINT_DIR)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # restore model tf Variables from disk
        saver.restore(sess, model_filepath)

        fd = {g.get_tensor_by_name("x:0"): test_data,
              g.get_tensor_by_name("keep_prob:0"): 1.0}  # no dropout when predicting

        # compute prediction
        test_labels = sess.run("predict/pred:0", feed_dict=fd)

        # load labels in a pandas DataFrame and write prediction file to disk
        submission = pd.DataFrame(data={"ImageId": (np.arange(test_labels.shape[0]) + 1), "Label": test_labels})

        current_dt = time.strftime("%Y%m%d-%H%M%S")
        submission.to_csv("prediction-%s.csv" % current_dt, index=False)


def main():

    # data, labels = load_data_from_file(TRAIN_SET, rows=50000)
    g = build_model(0.00001, size_conv1=6, size_conv2=16, size_fc1=120, size_fc2=84)
    #
    # current_dt = time.strftime("%Y%m%d-%H%M%S")
    # train_model(g, data, labels, steps=500, batch=100,
    #             run_name=current_dt + "-LeNet5-LR-1E-5-pass6",
    #             restore_checkpoint=True)

    make_prediction(g)

main()
