@ECHO OFF
setlocal
set PYTHONPATH=C:\Users\Timo\PycharmProjects\Kaggle\MNIST-tensorflow-cnn

SET ROOTDIR=C:\Users\Timo\PycharmProjects\Kaggle\MNIST-tensorflow-cnn

SET DATA_DIR=%ROOTDIR%\data\
SET EVENT_DIR=%ROOTDIR%\tf-event-files\
SET CHKP_DIR=%ROOTDIR%\tf-model-checkpoint\

SET LOAD_LAST_CHKP=True

python ./trainer/task.py --data-dir %DATA_DIR% --event-dir %EVENT_DIR% --chkp-dir %CHKP_DIR% --load_last_chkp %LOAD_LAST_CHKP%

endlocal

pause