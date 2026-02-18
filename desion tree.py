import math

# -----------------------------
# Fraud Detection Dataset
# -----------------------------
# Each record: [Amount, Location, Time, Fraud]
data = [
    ['Low', 'Local', 'Day', 'No'],
    ['Low', 'Foreign', 'Night', 'Yes'],
    ['High', 'Local', 'Night', 'Yes'],
    ['High', 'Foreign', 'Night', 'Yes'],
    ['Low', 'Local', 'Day', 'No'],
    ['High', 'Foreign', 'Day', 'No'],
    ['Low', 'Foreign', 'Night', 'Yes'],
    ['High', 'Local', 'Day', 'No']
]

features = ['Amount', 'Location', 'Time']


# -----------------------------
# Entropy Function
# -----------------------------
def entropy(dataset):
    total = len(dataset)
    count_yes = sum(1 for row in dataset if row[-1] == 'Yes')
    count_no = total - count_yes

    ent = 0
    for count in [count_yes, count_no]:
        if count != 0:
            p = count / total
            ent -= p * math.log2(p)

    return ent


# -----------------------------
# Information Gain
# -----------------------------
def information_gain(dataset, feature_index):
    total_entropy = entropy(dataset)
    total = len(dataset)

    values = set(row[feature_index] for row in dataset)
    weighted_entropy = 0

    for value in values:
        subset = [row for row in dataset if row[feature_index] == value]
        weighted_entropy += (len(subset) / total) * entropy(subset)

    return total_entropy - weighted_entropy


# -----------------------------
# ID3 Algorithm
# -----------------------------
def id3(dataset, feature_names):
    labels = [row[-1] for row in dataset]

    # If all same class
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    # If no features left
    if len(feature_names) == 0:
        return max(set(labels), key=labels.count)

    # Choose best feature
    gains = [information_gain(dataset, i) for i in range(len(feature_names))]
    best_index = gains.index(max(gains))
    best_feature = feature_names[best_index]

    tree = {best_feature: {}}

    values = set(row[best_index] for row in dataset)

    for value in values:
        subset = [row[:best_index] + row[best_index+1:]
                  for row in dataset if row[best_index] == value]

        remaining_features = feature_names[:best_index] + feature_names[best_index+1:]

        tree[best_feature][value] = id3(subset, remaining_features)

    return tree


# -----------------------------
# Build Decision Tree
# -----------------------------
decision_tree = id3(data, features)

print("Decision Tree:")
print(decision_tree)


# -----------------------------
# Classification Function
# -----------------------------
def classify(tree, sample):
    if not isinstance(tree, dict):
        return tree

    root = next(iter(tree))
    value = sample[root]

    return classify(tree[root][value], sample)


# -----------------------------
# New Transaction Classification
# -----------------------------
new_transaction = {
    'Amount': 'High',
    'Location': 'Foreign',
    'Time': 'Night'
}

result = classify(decision_tree, new_transaction)

print("\nNew Transaction:", new_transaction)
print("Fraud Prediction:", result)