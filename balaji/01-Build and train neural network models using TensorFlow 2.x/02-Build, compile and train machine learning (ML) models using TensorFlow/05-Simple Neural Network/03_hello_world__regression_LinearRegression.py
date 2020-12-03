"""
Using sklearn linear regression
"""
import numpy as np
from sklearn import linear_model

X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float).reshape(-1, 1)
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float).reshape(-1, 1)
regr = linear_model.LinearRegression()
regr.fit(X, y)
print(regr.predict(np.array([10]).reshape(-1, 1)))
