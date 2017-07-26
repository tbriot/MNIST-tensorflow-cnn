import tensorflow as tf
import os

import trainer.training_util as u


class EvalSessionHook(tf.train.SessionRunHook):

    def __init__(self,
                 graph,
                 every_n_iter,
                 checkpoint_dir,
                 event_dir):
        """

        :param graph:
        :param every_n_iter: runs an eval every n iterations if a new checkpoint is available
        :param checkpoint_dir:
        """
        self._every_n_iter = every_n_iter
        self._checkpoint_dir = checkpoint_dir
        self._graph = graph
        self._latest_checkpoint = None

        self._saver = tf.train.Saver()

        self._file_writer = tf.summary.FileWriter(
            os.path.join(event_dir, 'eval'), graph=graph)

        with graph.as_default():
            self._summary_op = tf.summary.merge_all(u.EVAL_SUMMARY_OP)

    def after_run(self,
                  run_context,
                  run_values):

        gs = tf.train.get_global_step()

        if gs % self._every_n_iter == 0:
            if self._new_chkpt():
                self._run_eval(gs)

    def _new_chkpt(self):
        latest = tf.train.latest_checkpoint(self._checkpoint_dir)
        is_new_chkpt = self._latest_checkpoint != latest
        if is_new_chkpt:
            self._latest_checkpoint = latest
        return is_new_chkpt

    def _run_eval(self, global_step):
        with tf.train.SingularMonitoredSession() as sess:
            self._saver.restore(sess, self._latest_checkpoint)
            while not sess.should_stop():
                summary = sess.run(self._summary_op)
                self._file_writer.add_summary(summary, global_step)
