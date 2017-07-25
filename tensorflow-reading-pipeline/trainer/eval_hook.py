import tensorflow as tf


class EvalSessionHook(tf.train.SessionRunHook):

    def __init__(self):
        pass

    def after_run(self,
                run_context,
                run_values):

        gs = tf.train.global_step(run_context.session, "global_step:0")

        if gs % 10 == 0:
            print("EvalSessionHook. Global step is", gs)
