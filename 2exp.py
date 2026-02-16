import numpy as np
import pandas as pd

# Read CSV file
data = pd.read_csv('enjoysport.csv')

# Separate attributes and target
concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

print("Training Data:\n", concepts)
print("\nTarget Values:\n", target)


def learn(concepts, target):

    # Initialize specific hypothesis
    specific_h = concepts[0].copy()

    # Initialize general hypothesis
    general_h = [["?" for _ in range(len(specific_h))]
                 for _ in range(len(specific_h))]

    print("\nInitial Specific Hypothesis:", specific_h)
    print("Initial General Hypothesis:", general_h)

    # Learning process
    for i in range(len(concepts)):

        if target[i].lower() == "yes":

            for j in range(len(specific_h)):

                if concepts[i][j] != specific_h[j]:
                    specific_h[j] = "?"
                    general_h[j][j] = "?"

        elif target[i].lower() == "no":

            for j in range(len(specific_h)):

                if concepts[i][j] != specific_h[j]:
                    general_h[j][j] = specific_h[j]
                else:
                    general_h[j][j] = "?"

        print(f"\nStep {i+1}")
        print("Specific Hypothesis:", specific_h)
        print("General Hypothesis:", general_h)

    # Remove fully general hypotheses
    general_h = [h for h in general_h if h != ["?"] * len(specific_h)]

    return specific_h, general_h


# Run algorithm
s_final, g_final = learn(concepts, target)

print("\nFinal Specific Hypothesis:")
print(s_final)

print("\nFinal General Hypothesis:")
print(g_final)
