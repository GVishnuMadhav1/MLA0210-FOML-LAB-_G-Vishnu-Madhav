import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load Data
print("Loading Mobile Price Data...")
try:
    df = pd.read_csv('mobile_price.csv')
    print(df.head())
except FileNotFoundError:
    print("mobile_price.csv not found!")
    exit()

# Features and Target
X = df.drop('price_range', axis=1)
y = df['price_range']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Classifier (Using SVM for mobile price range classification)
print("\nTraining SVM Classifier...")
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

# Predict
y_pred = svc.predict(X_test)

# Evaluate
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
