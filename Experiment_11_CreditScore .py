import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Using Decision Tree for classification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load Data
print("Loading Credit Score Data (credit_data.csv)...")
try:
    df = pd.read_csv('credit_data.csv')
    print(df.head())
except FileNotFoundError:
    print("credit_data.csv not found!")
    exit()

# Preprocessing
# Convert categorical strings to numbers
le = LabelEncoder()
for col in df.columns:
    if df[col].dtypes == 'object':
        df[col] = le.fit_transform(df[col])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Classifier (User didn't specify algo, using Decision Tree as it's good for credit scoring logic)
print("\nTraining Decision Tree Classifier...")
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
