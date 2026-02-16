import numpy as np
import pandas as pd

# Load Data
print("\nLoading Data from diabetes.csv...")
try:
    df = pd.read_csv('diabetes.csv')
    # Using a subset for demonstration speed and clarity if needed, but works with full
    X = df.iloc[:, :-1].values
    # Classes needed for classification. Diabetes dataset in sklearn is regression (quantitative target).
    # Let's binarize the target for a classification demo (High vs Low progression)
    y = df.iloc[:, -1].values
    y = (y > np.mean(y)).astype(int).reshape(-1, 1) # Simple binary classification
    
    # Normalize X
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
except FileNotFoundError:
    print("Error: diabetes.csv not found!")
    exit()

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize parameters
input_layer_neurons = X.shape[1]
hidden_layer_neurons = 10
output_neurons = 1

# Weights and Bias
# Random initialization
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))
wout = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

epoch = 1000  # Iterations
lr = 0.1 # Learning Rate

print("\nTraining Neural Network...")

for i in range(epoch):
    # Forward Prop
    hidden_layer_input1 = np.dot(X, wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    
    output_layer_input1 = np.dot(hiddenlayer_activations, wout)
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input)
    
    # Backpropagation
    E = y - output
    slope_output_layer = sigmoid_derivative(output)
    slope_hidden_layer = sigmoid_derivative(hiddenlayer_activations)
    
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    
    wout += hiddenlayer_activations.T.dot(d_output) * lr
    bout += np.sum(d_output, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hiddenlayer) * lr
    bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr
    
    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {np.mean(np.abs(E))}")

print("\nTraining Complete.")
print("Predicted Output (First 5 samples):")
print(output[:5])
print("Actual Output (First 5 samples):")
print(y[:5])
