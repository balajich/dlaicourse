"""
Preprocess the data before training the model
"""
import keras
from matplotlib import pyplot as plt

# load data set
(training_images, training_lables), (test_images, test_lables) = keras.datasets.fashion_mnist.load_data()


# Preprocess the data or  normalize between 0 to 1
# We are using SGD as optimizer  we must scale the pixel intensities down to the 0-1 range by dividing them by 255.0
training_images = training_images / 255.0
test_images = test_images / 255.0



# Build model
model = keras.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
                          keras.layers.Dense(1024, activation=keras.activations.relu),
                          keras.layers.Dense(10, activation=keras.activations.softmax)])

# Summary of the model
model.summary()
# compile
model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.sparse_categorical_crossentropy,
              metrics=[keras.metrics.sparse_categorical_accuracy])

# Train or fit
# model.fit(training_images, training_lables, epochs=10)
model.fit(training_images, training_lables, epochs=30)

# Predict
print(model.evaluate(test_images, test_lables))
