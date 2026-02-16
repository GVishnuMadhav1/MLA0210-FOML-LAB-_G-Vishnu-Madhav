import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Generate simple non-linear data
print("Generating Synthetic Data...")
np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 20)
y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, 20)

# Reshape for sklearn
X = X[:, np.newaxis]
y = y[:, np.newaxis]

# Linear Regression
print("\nTraining Linear Regression...")
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y, y_pred_lin)))
print("Linear Regression R2:", r2_score(y, y_pred_lin))

# Polynomial Regression (Degree 2)
print("\nTraining Polynomial Regression (Degree 2)...")
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

print("Polynomial Regression RMSE:", np.sqrt(mean_squared_error(y, y_pred_poly)))
print("Polynomial Regression R2:", r2_score(y, y_pred_poly))

# Visualization code (commented out for headless environment)
# plt.scatter(X, y, s=10)
# # Sort X for plotting line
# sort_axis = operator.itemgetter(0)
# sorted_zip = sorted(zip(X, y_pred_poly), key=sort_axis)
# X_sorted, y_poly_sorted = zip(*sorted_zip)
# plt.plot(X, y_pred_lin, color='r', label='Linear')
# plt.plot(X_sorted, y_poly_sorted, color='m', label='Polynomial')
# plt.legend()
# plt.show()
