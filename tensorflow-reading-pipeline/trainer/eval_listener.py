import tensorflow as tf
import os
import time

import trainer.training_util as u
import trainer.input_pipeline as ip
import trainer.model as m


class EvalCheckpointSaverListener(tf.train.CheckpointSaverListener):

    def __init__(self,
                 layers_layout,
                 eval_filename,
                 checkpoint_dir,
                 event_dir):

        self._checkpoint_dir = checkpoint_dir
        self._graph = tf.Graph()
        self._latest_checkpoint = None

        self._saver = tf.train.Saver()

        current_dt = time.strftime("%Y%m%d-%H%M%S")
        self._file_writer = tf.summary.FileWriter(
            os.path.join(event_dir, current_dt + '-eval'), graph=self._graph)

        with self._graph.as_default():
            features, labels = ip.input_pipeline(
                eval_filename,
                num_epochs=1,
                batch_size=50,
                shuffle=False)

            m.cnn_model(features, labels, layers_layout, keep_prob=1)
            self._summary_op = tf.summary.merge_all(u.EVAL_SUMMARY_OP)

    def after_save(self,
                   session,
                   global_step_value):

        # by default, the checkpoint saver calls back the 'after_save' method when global step = 1
        # we don't want to run an evaluation at this stage of the process
        if global_step_value > 1:

            self._latest_checkpoint = tf.train.latest_checkpoint(self._checkpoint_dir)

            print("[global step={}] New checkpoint created ({}). Running evaluation.".format(
                global_step_value,
                os.path.basename(self._latest_checkpoint))
            )

            eval_start_time = time.time()
            self._run_eval(global_step_value)
            eval_elapsed_time = time.time() - eval_start_time

            print("[global step={}] Evaluation completed in {:.0f} seconds.".format(global_step_value,
                                                                                    eval_elapsed_time))

    def _run_eval(self, global_step):
        with self._graph.as_default():
            with tf.train.SingularMonitoredSession() as sess:
                self._saver.restore(sess, self._latest_checkpoint)
                while not sess.should_stop():
                    summary = sess.run(self._summary_op)
                    self._file_writer.add_summary(summary, global_step)
