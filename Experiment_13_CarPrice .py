import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Load Data
print("Loading Car Data (car_data.csv)...")
try:
    df = pd.read_csv('car_data.csv')
    print(df.head())
except FileNotFoundError:
    print("car_data.csv not found!")
    exit()

# Preprocessing
# Encode categorical columns: Fuel_Type, Seller_Type, Transmission
le = LabelEncoder()
df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])
df['Seller_Type'] = le.fit_transform(df['Seller_Type'])
df['Transmission'] = le.fit_transform(df['Transmission'])

# Features & Target
# Price Prediction. Target: Selling_Price
X = df.drop(['Car_Name', 'Selling_Price'], axis=1) # Drop Name and Target
y = df['Selling_Price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Regressor (Using Random Forest for better accuracy than Linear)
print("\nTraining Random Forest Regressor...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Compare Logic
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nComparison (First 5):\n", comparison.head())
