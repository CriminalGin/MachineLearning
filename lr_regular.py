import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

X_train = [[6], [8], [10], [14], [18]]
y_train = [[7.5], [9.1], [13.2], [17.5], [19.3]]

X_test = [[6], [8], [11], [16]]
y_test = [[8.3], [12.5], [15.4], [18.6]]

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = lr_model.predict(xx.reshape(xx.shape[0], 1))
lr_score = lr_model.score(X_test, y_test)
plt.plot(xx, yy)


# poly = PolynomialFeatures(degree=5)
# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)
#
#
# ridge_model = Ridge(alpha=4, normalize=False)
# ridge_model.fit(X_train_poly, y_train)

