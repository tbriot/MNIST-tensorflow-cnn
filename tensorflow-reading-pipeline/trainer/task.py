import trainer.input_pipeline as ip
import trainer.model as m
import trainer.eval_listener as h
import trainer.update_lr_hook as lr
import tensorflow as tf
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run(train_filename, eval_filename, layers_layout, chkpt_dir, event_dir, training_program):

    features, labels = ip.input_pipeline(
        train_filename,
        num_epochs=3,
        batch_size=50,
        shuffle=False)

    train_op, global_step = m.cnn_model(features, labels, layers_layout, keep_prob=1)

    current_dt = time.strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.FileWriter(
        os.path.join(event_dir, current_dt + '-eval-train'),
        train_op.graph)
    summary_op = tf.summary.merge_all("debug")

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    chpkt_listener = h.EvalCheckpointSaverListener(
        layers_layout,
        eval_filename,
        chkpt_dir,
        event_dir
    )

    # create hook that updates the learning rate after each epoch
    # the learning rate decay is defined in the 'training_program'
    update_lr_hook = lr.UpdateLrSessionRunHook(
        training_program,
        tf.get_default_graph()
    )

    chkpt_hook = tf.train.CheckpointSaverHook(
        chkpt_dir,
        save_steps=500,
        listeners=[chpkt_listener]
    )

    scaf = tf.train.Scaffold(init_op=init_op)

    with tf.train.SingularMonitoredSession(hooks=[chkpt_hook, update_lr_hook], scaffold=scaf) as sess:
        while not sess.should_stop():
            _, summ, gs = sess.run([train_op, summary_op, global_step])
            file_writer.add_summary(summ, global_step=gs)
            # fc5, lab = sess.run(["full5/BiasAdd:0", "lab_run:0", train_op])[0:2]
            # print(fc5.dtype, lab.dtype)
            # feat = sess.run(["input_pipeline/feat:0", train_op])[0:1]
            # print(feat)

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

my_training_program = [{"lr": 1E-3, "epochs": 1},
                       {"lr": 5E-4, "epochs": 2}]

my_chkpt_dir = "C:/Users/Timo/PycharmProjects/Kaggle/MNIST/tf-model-checkpoint"
my_event_dir = "C:/Users/Timo/PycharmProjects/Kaggle/MNIST/tf-event-files"

my_train_file = "C:/Users/Timo/PycharmProjects/Kaggle/MNIST/data/train.csv"
my_eval_file = "C:/Users/Timo/PycharmProjects/Kaggle/MNIST/data/eval.csv"
run(my_train_file, my_eval_file, layers, my_chkpt_dir, my_event_dir, my_training_program)
