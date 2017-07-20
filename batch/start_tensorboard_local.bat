@ECHO OFF
setlocal

SET ROOTDIR=C:\Users\Timo\PycharmProjects\Kaggle\MNIST-tensorflow-cnn
SET EVENT_DIR=%ROOTDIR%\tf-event-files

tensorboard --logdir=%EVENT_DIR% --port 6006

endlocal
pause