import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load Data
print("Loading Housing Data...")
try:
    df = pd.read_csv('housing.csv')
    print(df.head())
except FileNotFoundError:
    print("Error: housing.csv not found!")
    exit()

# Preprocessing
# Convert categorical variables to numeric
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df.drop('Price', axis=1)
y = df['Price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Matrix
print("\nTraining Linear Regression for House Price Prediction...")
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\nModel Evaluation:")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

print("\nCoefficients:", model.coef_)
