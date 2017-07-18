import tensorflow as tf
from sklearn.model_selection import train_test_split
import time

# TODO add checkpoints


class CnnModel:

    IMAGE_H, IMAGE_W, IMAGE_C = 28, 28, 1  # height, width, channels (grayscale = just one channel)
    LABELS = 10

    LAYER_LAYOUT_TYPE_KEY = "type"
    LAYER_LAYOUT_TYPE_CONV, LAYER_LAYOUT_TYPE_FULL, LAYER_LAYOUT_DROP = "conv", "full", "drop"
    LAYER_LAYOUT_FILTER_KEY = "filter_size"
    LAYER_LAYOUT_DEPTH_KEY = "depth"
    LAYER_LAYOUT_MP_KEY = "mp_size"
    LAYER_LAYOUT_UNITS_KEY = "units"  # number of units in a fully connected layer
    LAYER_LAYOUT_ACT_KEY = "activation"  # boolean, True to add an activation function to the fully connected layer

    TRAIN_PROG_LR = "lr"  # training program learning rate
    TRAIN_PROG_EPOCHS = "epochs"  # training program epochs

    EVENT_DIR_BASE_NAME = "cnn-model"  # name of the directory storing event files produced by tf summaries

    def __init__(self, layers_layout):
        self._create_graph(layers_layout)

    # Example:
    # layers_layout = [{"type":"conv", "filter_size":5, "depth":6, "mp_size":2},
    #                  {"type":"conv", "filter_size":5, "depth":12, "mp_size":2},
    #                  {"type":"drop"},
    #                  {"type":"full", "units":32, "activation"=True}]
    #
    def _create_graph(self, layers_layout):
        self._graph = tf.Graph()

        with self._graph.as_default():
            self._create_placeholders()
            curr_layer = self._x

            # TODO can use curr_layer.shape[3] size instead ?
            curr_layer_depth = 1
            layer_num = 1  # layer number
            for _, layer in enumerate(layers_layout):
                # create convolutional layer
                if layer[self.LAYER_LAYOUT_TYPE_KEY] == self.LAYER_LAYOUT_TYPE_CONV:
                    curr_layer = self._create_conv_layer(layer_number=layer_num,
                                                         inputs=curr_layer,
                                                         in_channels=curr_layer_depth,
                                                         out_channels=layer[self.LAYER_LAYOUT_DEPTH_KEY],
                                                         filter_size=layer[self.LAYER_LAYOUT_FILTER_KEY],
                                                         mp_size=layer[self.LAYER_LAYOUT_MP_KEY])
                    curr_layer_depth = layer[self.LAYER_LAYOUT_DEPTH_KEY]
                    layer_num += 1

                # create fully connected layer
                elif layer[self.LAYER_LAYOUT_TYPE_KEY] == self.LAYER_LAYOUT_TYPE_FULL:
                    # flatten the previous layer output Tensor if not in 2-D
                    # for instance, if Tensor shape is [n, 7, 7, 64], reshapes it to [n, 7*7*64]
                    if len(curr_layer.shape) != 2:  # if not a 2-D Tensor
                        new_layer_shape = curr_layer.shape[1].value \
                                          * curr_layer.shape[2].value \
                                          * curr_layer.shape[3].value
                        curr_layer = tf.reshape(curr_layer, [-1, new_layer_shape], name="flatten")

                    curr_layer = tf.layers.dense(curr_layer,
                                                 units=layer[self.LAYER_LAYOUT_UNITS_KEY],
                                                 activation=tf.nn.relu if layer[self.LAYER_LAYOUT_ACT_KEY] else None,
                                                 use_bias=True,
                                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 bias_initializer=tf.zeros_initializer(),
                                                 name="full" + str(layer_num))
                    layer_num += 1

                # dropouts
                elif layer[self.LAYER_LAYOUT_TYPE_KEY] == self.LAYER_LAYOUT_DROP:
                    curr_layer = tf.nn.dropout(curr_layer, self._keep_prob, name="drop")

            logits = curr_layer  # the last layer output Tensor contains the logits

            pred_op = self._add_predict_op(logits)
            self._add_accuracy_sum(pred_op, self._y)

            loss_op = self._add_loss_op(logits, self._y)

            self._train_op = self._add_train_op(loss_op)

            # log weights of all trainable variables in the tf event file
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                tf.summary.histogram(var.name + "-sum", var, collections=["debug"])

    def _create_placeholders(self):
        with tf.name_scope("placeholder"):
            self._x = tf.placeholder(tf.float32, shape=[None, self.IMAGE_H, self.IMAGE_W, self.IMAGE_C], name="x")
            self._y = tf.placeholder(tf.float32, shape=[None, self.LABELS], name="y")
            # percentage of units to keep in the dropout layer (keep probability)
            self._keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob")

    @staticmethod
    def _create_conv_layer(layer_number,  # integer, conv layer # (1, 2, 3, ...)
                           inputs,  # Tensor, output of the previous layer
                           in_channels,  # integer, depth of the previous layer
                           out_channels,  # integer, depth of the output Tensor
                           filter_size=5,  # filter is a square
                           mp_size=2):  # max pool size
        with tf.name_scope("conv" + str(layer_number)):
            w = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], stddev=0.1),
                            name="w")
            b = tf.Variable(tf.constant(0.1, shape=[out_channels]), name="b")

            # conv op output Tensor shape is N*IMAGE_H*IMAGE_W*out_channels
            conv = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding="SAME", name="conv")
            relu = tf.nn.relu(conv + b, name="relu")
            # max pooling op output shape is N* IMAGE_H/2 * IMAGE_W/2 * out_channels
            return tf.nn.max_pool(relu, ksize=[1, mp_size, mp_size, 1],
                                  strides=[1, mp_size, mp_size, 1], padding="SAME")

    @staticmethod
    def _add_predict_op(logits):
        with tf.name_scope("predict"):
            return tf.argmax(logits, axis=1, name="pred")

    def _add_accuracy_sum(self, pred, labels):
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(pred, tf.argmax(labels, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self._accuracy_sum = tf.summary.scalar("accuracy", accuracy, collections=["xval", "debug"])

    @staticmethod
    def _add_loss_op(logits, labels):
        with tf.name_scope("loss"):
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                                  name="xent")

    def _add_train_op(self, loss):
        with tf.name_scope("train"):
            # learning rate (lr) is defined as a tf Variable
            # it will be updated during the training phase (1E-3, 5E-4, 1E-4, ...)
            self._lr = tf.Variable(initial_value=0.001, trainable=False, name="learning_rate")

            # log learning rate variable in the tf event file
            tf.summary.scalar("learning-rate", self._lr, collections=["debug"])

            # _global_step variable is a counter incremented at each call to the minimize() function
            self._global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")
            return tf.train.AdamOptimizer(self._lr).minimize(loss,  global_step=self._global_step, name="train_op")

    # Example:
    # training_program = [{"lr": 1E-3, "epochs":2},
    #                     {"lr": 5E-4, "epochs":3},
    #                     {"lr": 1E-4, "epochs":5}]
    #
    #
    def train(self, inputs, labels, training_program,
              event_file_dir,  # driectory tf summaries are written to
              batch=50,  # batch size
              val_set_size=0.1,  # cross validation set size, percentage
              keep_prob=0.5,  # keep probability, percentage of units kept while performing dropouts
              report_freq=100,  # report frequency, number of training steps after which summaries are written
              debug=False  # if debug mode is enabled, detailed summaries are logged to the tf event files
              ):
        # instantiate tf summaries file writer
        writer = tf.summary.FileWriter(self._build_event_filename(event_file_dir), graph=self._graph)

        print("Training started")
        print("Data set size id %d. Batch size is %d." % (inputs.shape[0], batch))

        print("Putting aside data for cross-validation")

        # put aside a cross-validation data set
        data_train, data_val, labels_train, labels_val = train_test_split(inputs, labels, test_size=val_set_size)
        print("Training set size is %d\nCross-validation set size is %d" % (data_train.shape[0], data_val.shape[0]))

        # compute number of calls to the tf training op per epochs
        # = records in training set // batch size
        steps_per_epoch = data_train.shape[0] // batch

        # define the feed dictionary for cross-validation
        fd_val = {self._x: data_val,
                  self._y: labels_val,
                  self._keep_prob: 1.0}  # no dropout when x-validating

        with self._graph.as_default():

            print("-----------------\n" +
                  "Session started")
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            # merge summaries for x-validation step
            sum_coll = "xval" if not debug else "debug"  # log more summaries if in debug mode
            sum_val = tf.summary.merge_all(sum_coll)

            # for each step of the training program...
            for i, program_step in enumerate(training_program):
                curr_epochs = program_step[self.TRAIN_PROG_EPOCHS]
                curr_lr = program_step[self.TRAIN_PROG_LR]

                print("Training step %d out of %d. Learning rate is %3.1E. Epochs is %d." %
                      (i+1,
                       len(training_program),
                       curr_lr,
                       curr_epochs))

                # update the learning rate
                sess.run(self._lr.assign(curr_lr))

                for epoch in range(curr_epochs):
                    for step in range(steps_per_epoch):
                        # get batch of random images and labels from the training set
                        batch_data, _, batch_labels, _ = train_test_split(data_train, labels_train, train_size=batch)

                        # create feed dictionary with batch data
                        fd = {self._x: batch_data,
                              self._y: batch_labels,
                              self._keep_prob: keep_prob}

                        # train the model and get the global step
                        _, gstep = sess.run([self._train_op, self._global_step], feed_dict=fd)

                        # every 100 steps compute accuracy on validation set
                        if gstep % report_freq == 0:
                            accu = sess.run(sum_val, feed_dict=fd_val)
                            writer.add_summary(accu, gstep)

                    print("- epoch %d out of %d completed" % (epoch + 1, curr_epochs))

    def _build_event_filename(self, dir_):
        current_dt = time.strftime("%Y%m%d-%H%M%S")
        return dir_ + current_dt + "-" + self.EVENT_DIR_BASE_NAME
