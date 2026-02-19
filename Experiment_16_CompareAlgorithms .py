import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Data
print("Loading Iris Data for Comparison...")
try:
    df = pd.read_csv('iris.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
except FileNotFoundError:
    print("iris.csv not found!")
    exit()

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Models
models = []
models.append(('Logistic Regression', LogisticRegression(max_iter=1000)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Decision Tree', DecisionTreeClassifier()))

# Evaluate each model
results = []
names = []
print("\nModel Comparison Results:")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append(acc)
    names.append(name)
    print(f"{name}: {acc:.4f}")

# Plotting
# plt.bar(names, results)
# plt.xlabel('Algorithms')
# plt.ylabel('Accuracy')
# plt.title('Algorithm Comparison')
# plt.show()
print("\n(Uncomment plotting code to visualize)")
