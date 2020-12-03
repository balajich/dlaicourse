import keras
from matplotlib import pyplot as plt

# load dataset
(training_images, training_lables), (test_images, test_lables) = keras.datasets.fashion_mnist.load_data()

#Sample size of train and test images
print(len(training_images))

# print image
print(training_images[0])
# View the image
plt.imshow(training_images[0])
# training_lables  or classes
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# check what first element in training label represent
print(class_names[training_lables[0]])  # it will be Ankle boot

# Preprocess the data - normalize
training_images = training_images / 255.0
test_images = test_images / 255.0



# Build model
model = keras.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
                          keras.layers.Dense(1024, activation=keras.activations.relu),
                          keras.layers.Dense(10, activation=keras.activations.softmax)])

# model = keras.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
#                           keras.layers.Dense(1024, activation=keras.activations.relu),
#                           keras.layers.Dense(512, activation=keras.activations.relu),
#                           keras.layers.Dense(256, activation=keras.activations.relu),
#                           keras.layers.Dense(128, activation=keras.activations.relu),
#                           keras.layers.Dense(10, activation=keras.activations.softmax)])
# Summary of the model
model.summary()
# compile
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy,
              metrics=[keras.metrics.sparse_categorical_accuracy])

# Train or fit
# model.fit(training_images, training_lables, epochs=10)
model.fit(training_images, training_lables, epochs=30)

# Predict
print(model.evaluate(test_images, test_lables))
