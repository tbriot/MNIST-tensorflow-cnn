@ECHO OFF
setlocal

SET ROOTDIR=gs://tbriot/machine-learning/MNIST
SET EVENT_DIR=%ROOTDIR%/tf-event-files

tensorboard --logdir=%EVENT_DIR% --port 6006

endlocal
pause