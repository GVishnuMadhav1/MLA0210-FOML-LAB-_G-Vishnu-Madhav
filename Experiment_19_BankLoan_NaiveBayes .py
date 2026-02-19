import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load Data
print("Loading Bank Loan Data...")
try:
    df = pd.read_csv('bank_loan.csv')
    print(df.head())
except FileNotFoundError:
    print("bank_loan.csv not found!")
    exit()

# Preprocessing
# Fill missing values if any (simple fill for demo)
df = df.fillna(method='ffill')

# Encode categorical variables
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes
print("\nTraining Naive Bayes for Bank Loan Prediction...")
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predict
y_pred = nb.predict(X_test)

# Evaluate
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
