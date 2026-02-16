import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

# Load Data
print("\nLoading Data from tennis.csv...")
try:
    data = pd.read_csv('tennis.csv')
    print(data.head())
except FileNotFoundError:
    print("Error: tennis.csv not found!")
    exit()

# Preprocessing: Encode categorical variables
le = LabelEncoder()
data_encoded = data.apply(le.fit_transform)

X = data_encoded.iloc[:, :-1]
y = data_encoded.iloc[:, -1]

print("\nEncoded Data:")
print(data_encoded.head())

# ID3 Algorithm uses Information Gain (entropy)
# Sklearn's DecisionTreeClassifier uses CART by default, but with criterion='entropy' it mimics ID3/C4.5 behavior for splitting
# Note: Sklearn implementation doesn't support categorical data directly, hence we encoded it.
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

print("\nDecision Tree Rules:\n")
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)

# Prediction for a new sample
# Example: Outlook=Sunny, Temperature=Cool, Humidity=High, Wind=Strong
# We need to encode this new sample using the same mappings
# Since standard LabelEncoder transforms based on alphabetical order of seen labels:
# Outlook: Overcast=0, Rain=1, Sunny=2
# Temp: Cool=0, Hot=1, Mild=2
# Humidity: High=0, Normal=1
# Wind: Strong=0, Weak=1
# (Note: In a real robust system, we would save the encoders. Here we manually infer or use the same encoder object if possible per column)

new_sample = [[2, 0, 0, 0]] # Sunny, Cool, High, Strong
pred = clf.predict(new_sample)
pred_class = le.fit(data.iloc[:, -1]).inverse_transform(pred) # Inverse transform using target column

print(f"\nPrediction for new sample {new_sample} (Sunny, Cool, High, Strong): {pred_class[0]}")
