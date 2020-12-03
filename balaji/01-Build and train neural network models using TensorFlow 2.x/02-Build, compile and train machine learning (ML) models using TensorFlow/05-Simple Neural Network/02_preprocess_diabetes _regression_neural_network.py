"""
Preprocess the data before training the model
"""
import keras

from sklearn import datasets

# load data set and split into train and test
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Build model
model = keras.Sequential([keras.layers.Dense(units=10, input_dim=10, activation=keras.activations.relu),
                          keras.layers.Dense(units=1, activation=keras.activations.relu)])

# Summary of the model
model.summary()
# compile
model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.mean_squared_error,
              metrics=[keras.metrics.mean_squared_error])

# Train or fit
model.fit(diabetes_X_train, diabetes_y_train, epochs=30)

# Predict
print(model.evaluate(diabetes_X_test, diabetes_y_test))
