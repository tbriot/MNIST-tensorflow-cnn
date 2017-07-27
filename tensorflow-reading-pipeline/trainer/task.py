import trainer.input_pipeline as ip
import trainer.model as m
import trainer.eval_listener as h
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run(train_filename, eval_filename, layers_layout, chkpt_dir, event_dir):
    features, labels = ip.input_pipeline(
        train_filename,
        num_epochs=2,
        batch_size=50,
        shuffle=True)

    train_op, global_step = m.cnn_model(features, labels, layers_layout, keep_prob=0.5)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    chpkt_listener = h.EvalCheckpointSaverListener(
        layers_layout,
        eval_filename,
        chkpt_dir,
        event_dir
    )

    chkpt_hook = tf.train.CheckpointSaverHook(
        chkpt_dir,
        save_steps=500,
        listeners=[chpkt_listener]
    )

    scaf = tf.train.Scaffold(init_op=init_op)

    with tf.train.SingularMonitoredSession(hooks=[chkpt_hook], scaffold=scaf) as sess:
        while not sess.should_stop():
            sess.run(train_op)

    """
    # Create a session for running operations in the Graph.
    sess = tf.Session()
    # Initialize variables
    sess.run(init_op)
    # sess.run(tf.initialize_all_variables())

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # Run training steps or whatever
            _, gs = sess.run([train_op, global_step])
            print("Global step is", gs)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

        # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    """


# define our CNN model layout
layers = [{"type": "conv", "filter_size": 5, "depth": 6, "mp_size": 2},
          {"type": "conv", "filter_size": 5, "depth": 16, "mp_size": 2},
          {"type": "drop"},
          {"type": "full", "units": 120, "activation": True},
          {"type": "full", "units": 84, "activation": True},
          {"type": "full", "units": 10, "activation": False}]

chkpt_dir = "C:/Users/Timo/PycharmProjects/Kaggle/MNIST/tf-model-checkpoint"
event_dir = "C:/Users/Timo/PycharmProjects/Kaggle/MNIST/tf-event-files"

train_file = "C:/Users/Timo/PycharmProjects/Kaggle/MNIST/data/train.csv"
eval_file = "C:/Users/Timo/PycharmProjects/Kaggle/MNIST/data/eval.csv"
run(train_file, eval_file, layers, chkpt_dir, event_dir)
