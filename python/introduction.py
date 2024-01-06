# Uncertainty quantification for regression problem
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from mapie.regression import MapieRegressor

import matplotlib.pyplot as plt

#X, y = make_regression(n_samples=1_000, n_features=1, noise=10, random_state=42)
n_samples = 100
X = np.linspace(0,20,n_samples)
y_base = X * np.sin(X)
y = y_base + np.random.randn(n_samples)
X = X.reshape(-1,1)
# plt.plot(x,y,'.')
# plt.plot(x,y_base)
# plt.show()

#y =  (np.sin(X.T) * y).T.flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

#regressor = LinearRegression()
regressor = RandomForestRegressor(max_depth=10, random_state=0)

mapie_regressor = MapieRegressor(estimator=regressor, method='plus', cv=5)

mapie_regressor = mapie_regressor.fit(X_train, y_train)
y_pred, y_pis = mapie_regressor.predict(X_test, alpha=[0.05, 0.3])


# plot it
order = np.argsort(X_test.flatten())
plt.fill_between(X_test[order,0], y_pis[order, 0, 0], y_pis[order, 1, 0],
                 color='skyblue', alpha=0.5, label='95% prediction interval' )
plt.fill_between(X_test[order,0], y_pis[order, 0, 1], y_pis[order, 1, 1],
                 color='gold', alpha=0.5, label='70% prediction interval' )

plt.plot(X_test[order,0], y_pred[order], '.r', label='mean prediction')
plt.scatter(X_test[order,0], y_test[order], color='blue', s=10, label='Test points')
plt.scatter(X_train[order,0], y_train[order], color='green', s=10, label='Train points')
plt.plot(X.flatten(), y_base, '-k', label=True)
plt.legend()
plt.show()
