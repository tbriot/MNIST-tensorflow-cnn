import tensorflow as tf
import trainer.training_util as u

IMAGE_H, IMAGE_W, IMAGE_C = 28, 28, 1  # height, width, channels (grayscale = just one channel)
LABELS = 10

LAYER_LAYOUT_TYPE_KEY = "type"
LAYER_LAYOUT_TYPE_CONV, LAYER_LAYOUT_TYPE_FULL, LAYER_LAYOUT_DROP = "conv", "full", "drop"
LAYER_LAYOUT_FILTER_KEY = "filter_size"
LAYER_LAYOUT_DEPTH_KEY = "depth"
LAYER_LAYOUT_MP_KEY = "mp_size"
LAYER_LAYOUT_UNITS_KEY = "units"  # number of units in a fully connected layer
LAYER_LAYOUT_ACT_KEY = "activation"  # boolean, True to add an activation function to the fully connected layer


# Example:
# layers_layout = [{"type":"conv", "filter_size":5, "depth":6, "mp_size":2},
#                  {"type":"conv", "filter_size":5, "depth":12, "mp_size":2},
#                  {"type":"drop"},
#                  {"type":"full", "units":32, "activation"=True}]
#
def cnn_model(features,
              labels,
              layers_layout,
              keep_prob):

    with features.graph.as_default():

        # percentage of units to keep in the dropout layer (keep probability)
        keep_prob = tf.constant(keep_prob, tf.float32, shape=[], name="keep_prob")

        # global_step variable is a counter incremented at each call to the minimize() function
        global_step = tf.train.create_global_step()

        curr_layer = features
        curr_layer_depth = 1

        layer_num = 1  # layer number
        for _, layer in enumerate(layers_layout):
            # create convolutional layer
            if layer[LAYER_LAYOUT_TYPE_KEY] == LAYER_LAYOUT_TYPE_CONV:
                curr_layer = _create_conv_layer(layer_number=layer_num,
                                                inputs=curr_layer,
                                                in_channels=curr_layer_depth,
                                                out_channels=layer[LAYER_LAYOUT_DEPTH_KEY],
                                                filter_size=layer[LAYER_LAYOUT_FILTER_KEY],
                                                mp_size=layer[LAYER_LAYOUT_MP_KEY])
                curr_layer_depth = layer[LAYER_LAYOUT_DEPTH_KEY]
                layer_num += 1

            # create fully connected layer
            elif layer[LAYER_LAYOUT_TYPE_KEY] == LAYER_LAYOUT_TYPE_FULL:
                # flatten the previous layer output Tensor if not in 2-D
                # for instance, if Tensor shape is [n, 7, 7, 64], reshapes it to [n, 7*7*64]
                if len(curr_layer.shape) != 2:  # if not a 2-D Tensor
                    new_layer_shape = curr_layer.shape[1].value \
                                      * curr_layer.shape[2].value \
                                      * curr_layer.shape[3].value
                    curr_layer = tf.reshape(curr_layer, [-1, new_layer_shape], name="flatten")

                curr_layer = tf.layers.dense(curr_layer,
                                             units=layer[LAYER_LAYOUT_UNITS_KEY],
                                             activation=tf.nn.relu if layer[LAYER_LAYOUT_ACT_KEY] else None,
                                             use_bias=True,
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                             bias_initializer=tf.zeros_initializer(),
                                             name="full" + str(layer_num))
                layer_num += 1

            # dropouts
            elif layer[LAYER_LAYOUT_TYPE_KEY] == LAYER_LAYOUT_DROP:
                curr_layer = tf.nn.dropout(curr_layer, keep_prob, name="drop")

        logits = curr_layer  # the last layer output Tensor contains the logits

        pred_op = tf.argmax(logits, axis=1, name="predict")
        _add_accuracy_ops(pred_op, labels)

        loss_op = _add_loss_op(logits, labels)
        train_op = _add_train_op(loss_op, global_step)

        # log weights of all trainable variables in the tf event file
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            tf.summary.histogram(var.name + "-sum", var, collections=["debug"])

        return train_op, global_step


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


def _add_accuracy_ops(pred, labels):
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(pred, tf.argmax(labels, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy, collections=[u.EVAL_SUMMARY_OP, "debug"])


def _add_loss_op(logits, labels):
    with tf.name_scope("loss"):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                              name="xent")


def _add_train_op(loss, global_step):
    with tf.name_scope("train"):
        # learning rate (lr) is defined as a tf Variable
        # it will be updated during the training phase (1E-3, 5E-4, 1E-4, ...)
        lr = tf.Variable(initial_value=0.001, trainable=False, name="learning_rate")

        # log learning rate variable in the tf event file
        tf.summary.scalar("learning-rate", lr, collections=["debug"])

        optim = tf.train.AdamOptimizer(lr)
        return optim.minimize(loss, global_step=global_step, name="train_op")
