# y = 1 * x_0 + 2 * x_1 + 3
import keras
import numpy as np

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
model = keras.Sequential([keras.layers.Dense(units=2, input_dim=2, activation=keras.activations.relu,
                                             kernel_initializer=keras.initializers.random_uniform),
                          keras.layers.Dense(units=1)])
model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.mean_squared_error,
              metrics=[keras.metrics.Accuracy()])
model.summary()
model.fit(X, y, epochs=100)
print(model.predict(np.array([[3, 5]])))
