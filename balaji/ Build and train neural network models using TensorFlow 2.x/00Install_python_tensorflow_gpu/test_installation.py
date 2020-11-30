import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
print(keras.__version__)
# test whether tensorflow is using GPU are not
print(tf.test.is_gpu_available())
# Get my GPU device name
print(tf.test.gpu_device_name())
