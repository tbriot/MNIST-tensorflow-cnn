import tensorflow as tf

# integer, example queue capacity as a multiple of the batch size
# e.g. '10' creates an example queue size 10 times the size of a batch
EXAMPLE_QUEUE_CAPACITY_MULTI = 10
# between 0 and 1, min elements in the shuffle queue as a percentage of the queue capacity
# e.g. '0.4' creates an example queue which is always at least 40% full
EXAMPLE_QUEUE_MIN_THRESHOLD = 0.4
# number of threads enqueing the example queue
EXAMPLE_QUEUE_NUM_THREADS = 1


def preprocess_features(features, labels):
    """
    :param features: 2-D Tensor. shape is [N, 784]  (H*W*C = 28*28*1 = 784)
    :return: 4-D Tensor. shape is [N, 28, 28, 1]
    """
    # TODO tf.nn.batch_normalization ?
    features = tf.reshape(features, [-1, 28, 28, 1])
    labels = tf.one_hot(tf.cast(labels, tf.int32), 10)

    return features, labels


def read_file(
        filename_queue,
        batch_size,
        skip_header_lines=True):

        reader = tf.TextLineReader(skip_header_lines=skip_header_lines)
        _, rows = reader.read_up_to(filename_queue, num_records=batch_size)

        records_defaults = [[]] * 785  # all columns are required

        # first column contains the label (digit from '0' to '9')
        # followed by 784 columns, one for each pixel (28x28 image)
        features = tf.decode_csv(rows, record_defaults=records_defaults)
        labels = features.pop(0)

        return preprocess_features(features, labels)


def batch(
        features,
        labels,
        batch_size,
        shuffle=True):

    capacity = batch_size * EXAMPLE_QUEUE_CAPACITY_MULTI
    min_after_dequeue = int(capacity * EXAMPLE_QUEUE_MIN_THRESHOLD)

    if shuffle:
        features_batch, label_batch = tf.train.shuffle_batch(
            [features, labels],
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=True)
    else:
        features_batch, label_batch = tf.train.batch(
            [features, labels],
            batch_size=batch_size,
            num_threads=EXAMPLE_QUEUE_NUM_THREADS,
            capacity=capacity,
            enqueue_many=True,
            allow_smaller_final_batch=True)

    return features_batch, label_batch


def input_pipeline(
        filename,
        num_epochs=1,
        batch_size=50,
        shuffle=True
        ):

    with tf.name_scope('input_pipeline'):
        # create filename queue
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
        # read, parse the file and preprocess the input data
        features, labels = read_file(filename_queue, batch_size)
        # create another queue to batch together examples
        return batch(features, labels, batch_size, shuffle)
