"""
Predicting price of multi storey  building using Neural Networks
"""
import keras
import numpy as np

X = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)  # no of floors
y = np.array([50, 100, 150, 200, 250, 300, 350], dtype=float)  # cost of building
# Build a Sequential Neural network with one neuron and input shape as 1 value
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# Compile with guess function (optimizer) as STOCHASTIC GRADIENT DESCENT
# loss function as MEAN SQUARED ERROR
model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.mean_squared_error)
