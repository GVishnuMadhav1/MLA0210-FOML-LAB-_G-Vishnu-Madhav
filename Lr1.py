# ----------------------------------------
# Logistic Regression for Fraud Detection
# Without CSV and without sklearn
# ----------------------------------------

import numpy as np

# Encoded dataset
# Amount: Low=1, Medium=2, High=3
# Location: Local=1, Foreign=2
# Time: Day=1, Night=2
# Fraud: No=0, Yes=1

data = np.array([
    [1, 1, 1, 0],
    [1, 2, 2, 1],
    [3, 1, 2, 1],
    [3, 2, 2, 1],
    [1, 1, 1, 0],
    [3, 2, 1, 0],
    [1, 2, 2, 1],
    [3, 1, 1, 0]
])

# Split features and labels
X = data[:, :-1]
y = data[:, -1]

# Add bias column (1s)
X = np.c_[np.ones(X.shape[0]), X]

# ----------------------------------------
# Sigmoid function
# ----------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ----------------------------------------
# Train logistic regression using gradient descent
# ----------------------------------------
weights = np.zeros(X.shape[1])
learning_rate = 0.1
epochs = 1000

for _ in range(epochs):
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    
    # Gradient descent update
    gradient = np.dot(X.T, (predictions - y)) / len(y)
    weights -= learning_rate * gradient

# ----------------------------------------
# Predict new transaction
# Example: High amount, Foreign, Night
# ----------------------------------------
new_transaction = np.array([1, 3, 2, 2])  # bias + features

z = np.dot(new_transaction, weights)
probability = sigmoid(z)

# ----------------------------------------
# Output
# ----------------------------------------
print("Fraud Probability:", round(probability, 4))

if probability >= 0.5:
    print("Prediction: Fraudulent Transaction")
else:
    print("Prediction: Normal Transaction")