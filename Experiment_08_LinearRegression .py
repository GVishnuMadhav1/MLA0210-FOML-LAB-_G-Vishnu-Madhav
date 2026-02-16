import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
# Using 'housing.csv' or creating a simple one. Housing csv was created in planning.
print("Loading Housing Data...")
try:
    df = pd.read_csv('housing.csv')
    # Preprocessing: Select numerical features directly or encode
    # For simplicity, let's select Area (feature) vs Price (target) for Simple Linear Regression
    X = df[['Area']].values
    y = df['Price'].values
except FileNotFoundError:
    print("housing.csv not found, generating dummy data...")
    X = np.array([[1000], [1500], [2000], [2500], [3000]])
    y = np.array([150000, 225000, 300000, 375000, 450000])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
print("\nTraining Linear Regression Model...")
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)

# Evaluate
print("\nCoefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plotting (Optional, nice for demo)
# plt.scatter(X, y, color='blue')
# plt.plot(X, regressor.predict(X), color='red')
# plt.show()
print("\nNote: Uncomment plotting code to visualize if running locally with UI.")
