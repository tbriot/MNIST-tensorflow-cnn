@ECHO OFF
setlocal

REM ------------------
REM GCS args
REM ------------------
SET JOB_NAME=MNIST_JOB_7
SET TRAINER_PACKAGE_PATH=C:\Users\Timo\PycharmProjects\Kaggle\MNIST-tensorflow-cnn\trainer
SET MAIN_TRAINER_MODULE=trainer.task
SET JOB_DIR=gs://tbriot/machine-learning/MNIST/output/%JOB_NAME%
SET REGION=us-east1
SET RUNTIME_VERSION=1.2

REM ------------------
REM Custom trainer args
REM ------------------
SET ROOTDIR=gs://tbriot/machine-learning/MNIST
SET DATA_DIR=%ROOTDIR%/data/
SET EVENT_DIR=%ROOTDIR%/tf-event-files/
SET CHKP_DIR=%ROOTDIR%/tf-model-checkpoint/
SET LOAD_LAST_CHKP=False

gcloud ml-engine jobs submit training %JOB_NAME% ^
--job-dir %JOB_DIR% ^
--package-path %TRAINER_PACKAGE_PATH% ^
--module-name %MAIN_TRAINER_MODULE% ^
--region %REGION% ^
--runtime-version %RUNTIME_VERSION% ^
-- ^
--data-dir %DATA_DIR% ^
--event-dir %EVENT_DIR% ^
--chkp-dir %CHKP_DIR% ^
--load_last_chkp %LOAD_LAST_CHKP%

endlocal
pause