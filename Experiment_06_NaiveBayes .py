import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Load Data
print("Loading Iris Data for Naive Bayes...")
df = pd.read_csv('iris.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Naive Bayes Classifier
print("Training Naive Bayes Model...")
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)

# Results
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
print("\nAccuracy Score:", ac)
