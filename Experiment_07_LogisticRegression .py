import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Data
# We will use the 'diabetes.csv' but with a binary target for classification
# Or we can create a simple binary dataset. Let's use diabetes and binarize the target again.
print("Loading Data...")
df = pd.read_csv('diabetes.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# Binarize target: 1 if above mean (High Risk), 0 if below (Low Risk)
import numpy as np
y_binary = (y > np.mean(y)).astype(int)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.25, random_state=42)

# Logistic Regression
print("\nTraining Logistic Regression Model...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict
y_pred = log_reg.predict(X_test)

# Evaluate
print("\nResults:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
