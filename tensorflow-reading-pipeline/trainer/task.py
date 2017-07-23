import trainer.input_pipeline as ip
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run(filename):

    features, label = ip.input_pipeline(
        filename,
        num_epochs=2,
        batch_size=50,
        shuffle=True)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Create a session for running operations in the Graph.
    sess = tf.Session()
    # Initialize variables
    sess.run(init_op)
    # sess.run(tf.initialize_all_variables())

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    i = 0
    try:
        while (not coord.should_stop()) and (i < 1):
            # Run training steps or whatever
            print(sess.run([features, label]))
            i += 1

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

        # Wait for threads to finish.
    coord.join(threads)
    sess.close()

train_file = "C:/Users/Timo/PycharmProjects/Kaggle/MNIST/data/train.csv"
run(train_file)
