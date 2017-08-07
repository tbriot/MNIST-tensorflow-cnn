EVAL_SUMMARY_OP = 'EVAL_SUMMARY_OP'
READER_WORK_UNIT_CNT = 'READER_WU'  # counter to track work unit dequeued by the input pipeline text reader
LEARNING_RATE_OP = 'LR_OP'


def get_lr_op(graph):
    return graph.get_collection(LEARNING_RATE_OP)[0]


def get_reader_wu_op(graph):
    return graph.get_collection(READER_WORK_UNIT_CNT)[0]


def get_epochs(training_program):
    num_epochs = 0
    for step in training_program:
        num_epochs += step['epochs']
    return num_epochs
