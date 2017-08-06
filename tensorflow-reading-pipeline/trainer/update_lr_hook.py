import trainer.training_util as u
import tensorflow as tf

"""
Update the learning rate after completion of an epoch
"""


class UpdateLrSessionRunHook(tf.train.SessionRunHook):
    def __init__(self,
                 training_program,
                 graph):
        self._lr_op = u.get_lr_op(graph)
        self._last_run_epoch_op = u.get_reader_wu_op(graph)

        self._curr_epoch = 0
        self._lr_list = self._convert_program_to_list(training_program)

        self.curr_lr = self._lr_list.pop(0)
        print("Initializing learning rate to {:.1E}".format(self.curr_lr))

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self._last_run_epoch_op, feed_dict={self._lr_op: self.curr_lr})

    def after_run(self,
                  run_context,
                  run_values):
        last_run_epoch = run_values[0]  # get the epoch of the last run

        if last_run_epoch > self._curr_epoch and len(self._lr_list) > 0:
            print("New epoch starting. Setting learning rate to {:.1E}".format(self._lr_list[0]))
            self.curr_lr = self._lr_list.pop(0)
            self._curr_epoch = last_run_epoch

    @staticmethod
    def _convert_program_to_list(training_program):
        """

        :param training_program:

        Example = [{"lr": 1E-3, "epochs":1},
                   {"lr": 5E-4, "epochs":2},
                   {"lr": 1E-4, "epochs":3}]
        :return:

        Example: [1E-3, 5E-4, 5E-4, 1E-4, 1E-4, 1E-4]
        """
        output_list = []

        for step in training_program:
            output_list = output_list + [step['lr']] * step['epochs']

        return output_list
