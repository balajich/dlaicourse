import numpy as np
import keras
# Build
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compile
model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.mean_squared_error)
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
# Train
model.fit(xs, ys, epochs=500)
# Predict
print(model.predict([10]))
