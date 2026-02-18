import math
from collections import Counter

# --------------------------------
# Fraud Detection Dataset
# Encoded Numerically
# Amount: Low=1, Medium=2, High=3
# Location: Local=1, Foreign=2
# Time: Day=1, Night=2
# Fraud: No=0, Yes=1
# --------------------------------

data = [
    [1, 1, 1, 0],  # Low, Local, Day → No
    [1, 2, 2, 1],  # Low, Foreign, Night → Yes
    [3, 1, 2, 1],  # High, Local, Night → Yes
    [3, 2, 2, 1],  # High, Foreign, Night → Yes
    [1, 1, 1, 0],
    [3, 2, 1, 0],
    [1, 2, 2, 1],
    [3, 1, 1, 0]
]

k = 3

# --------------------------------
# Distance Function (Euclidean)
# --------------------------------
def euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1) - 1):  # exclude class label
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)

# --------------------------------
# KNN Classification Function
# --------------------------------
def knn_classify(dataset, new_point, k):
    distances = []

    for row in dataset:
        dist = euclidean_distance(row, new_point)
        distances.append((dist, row[-1]))

    distances.sort(key=lambda x: x[0])

    neighbors = distances[:k]

    labels = [label for _, label in neighbors]

    prediction = Counter(labels).most_common(1)[0][0]

    return prediction

# --------------------------------
# New Transaction to Classify
# Example: High, Foreign, Night
# Encoded as [3, 2, 2]
# --------------------------------
new_transaction = [3, 2, 2, None]

result = knn_classify(data, new_transaction, k)

# --------------------------------
# Output Result
# --------------------------------
print("New Transaction (Encoded):", new_transaction[:-1])

if result == 1:
    print("Fraud Prediction: Yes (Fraudulent Transaction)")
else:
    print("Fraud Prediction: No (Normal Transaction)")