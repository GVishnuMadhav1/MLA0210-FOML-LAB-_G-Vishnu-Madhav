import math
from collections import Counter

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    distance = 0
    for i in range(len(p1)):
        distance += (p1[i] - p2[i]) ** 2
    return math.sqrt(distance)

# KNN Algorithm
def knn(train_data, train_labels, test_point, k):
    distances = []

    # Calculate distance from test point to all training points
    for i in range(len(train_data)):
        dist = euclidean_distance(train_data[i], test_point)
        distances.append((dist, train_labels[i]))

    # Sort distances
    distances.sort(key=lambda x: x[0])

    # Get k nearest neighbors
    k_nearest_labels = [label for _, label in distances[:k]]

    # Majority vote
    prediction = Counter(k_nearest_labels).most_common(1)[0][0]
    return prediction

# Training data
train_data = [
    [1, 2],
    [2, 3],
    [3, 4],
    [6, 7],
    [7, 8],
    [8, 9]
]

# Class labels
train_labels = ['A', 'A', 'A', 'B', 'B', 'B']

# Test data point
test_point = [5, 5]

# Value of K
k = 3

# Prediction
result = knn(train_data, train_labels, test_point, k)
print("Predicted Class:", result)
