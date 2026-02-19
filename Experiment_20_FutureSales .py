import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load Data
print("Loading Sales Data...")
try:
    df = pd.read_csv('sales_data.csv')
    print(df.head())
except FileNotFoundError:
    print("sales_data.csv not found!")
    exit()

# Preprocessing
# Convert Date to numeric (ordinal)
df['Date'] = pd.to_datetime(df['Date'])
df['Date_Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
# Encode Item
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Item'] = le.fit_transform(df['Item'])

X = df[['Date_Ordinal', 'Item']]
y = df['Sales']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
print("\nTraining Linear Regression for Future Sales Prediction...")
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Future Sales
# Example: Predict for Item A (0) on next day
last_date = df['Date'].max()
future_date = last_date + pd.Timedelta(days=1)
future_date_ordinal = future_date.toordinal()

future_X = np.array([[future_date_ordinal, 0]]) # Item 0
future_pred = model.predict(future_X)

print(f"\nPredicted Sales for Item A on {future_date.date()}: {future_pred[0]:.2f}")

# Model Score
print("Model R2 Score:", model.score(X_test, y_test))
