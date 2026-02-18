# ----------------------------------------
# Naïve Bayes Classifier for Fraud Detection
# Dataset is defined inside the program
# ----------------------------------------

# Encoded Dataset
# Amount: Low=1, Medium=2, High=3
# Location: Local=1, Foreign=2
# Time: Day=1, Night=2
# Fraud: No=0, Yes=1

data = [
    [1, 1, 1, 0],
    [1, 2, 2, 1],
    [3, 1, 2, 1],
    [3, 2, 2, 1],
    [1, 1, 1, 0],
    [3, 2, 1, 0],
    [1, 2, 2, 1],
    [3, 1, 1, 0]
]

# ----------------------------------------
# Separate data by class
# ----------------------------------------
def separate_by_class(dataset):
    separated = {}
    for row in dataset:
        label = row[-1]
        if label not in separated:
            separated[label] = []
        separated[label].append(row)
    return separated

# ----------------------------------------
# Calculate probability with Laplace smoothing
# ----------------------------------------
def calculate_probability(class_data, index, value, total_values):
    count = sum(1 for row in class_data if row[index] == value)
    return (count + 1) / (len(class_data) + total_values)

# ----------------------------------------
# Naïve Bayes prediction
# ----------------------------------------
def naive_bayes_predict(dataset, new_data):
    separated = separate_by_class(dataset)
    total_rows = len(dataset)

    probabilities = {}

    for class_value, class_data in separated.items():
        # Prior probability P(class)
        prior = len(class_data) / total_rows
        prob = prior

        print(f"\nClass {class_value} prior probability:", prior)

        # For each feature
        for i in range(len(new_data)):
            # Possible values per feature
            if i == 0:
                total_values = 3
            else:
                total_values = 2

            cond_prob = calculate_probability(class_data, i, new_data[i], total_values)
            prob *= cond_prob

            print(f"P(feature {i+1}={new_data[i]} | class {class_value}) =", cond_prob)

        probabilities[class_value] = prob
        print("Final probability for class", class_value, "=", prob)

    # Choose class with highest probability
    prediction = max(probabilities, key=probabilities.get)

    return prediction, probabilities

# ----------------------------------------
# New transaction to classify
# Example: High amount, Foreign, Night
# Encoded as [3, 2, 2]
# ----------------------------------------
new_transaction = [3, 2, 2]

prediction, probs = naive_bayes_predict(data, new_transaction)

# ----------------------------------------
# Output result
# ----------------------------------------
print("\nNew Transaction:", new_transaction)

if prediction == 1:
    print("Prediction: Fraudulent Transaction")
else:
    print("Prediction: Normal Transaction")