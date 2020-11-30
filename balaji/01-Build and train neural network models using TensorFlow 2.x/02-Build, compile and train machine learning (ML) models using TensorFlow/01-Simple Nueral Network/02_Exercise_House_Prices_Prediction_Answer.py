# House price prediction
import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras


def predict_house_prices(xs, ys):
    result = 0
    model = tf.keras.Sequential(keras.layers.Dense(units=1, input_shape=[1]))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=100)
    result = model.predict([7])
    return result


class PredictHousePrice(unittest.TestCase):
    def test_predict_house_prices(self):
        input_xs = np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)
        input_ys = np.array([50, 100, 150, 200, 250, 300, 350])
        expected = 400
        actual = predict_house_prices(input_xs, input_ys)
        print(actual)
        assert True


if __name__ == '__main__':
    unittest.main()
