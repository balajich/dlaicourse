import numpy as np
import tensorflow as tf
from tensorflow import  keras
import matplotlib.pyplot as plt



#load dataset
(training_images, training_lables), (test_images, test_lables)=tf.keras.datasets.fashion_mnist.load_data()
#print image
print(training_images[0])
# View the image
plt.imshow(training_images[0])



