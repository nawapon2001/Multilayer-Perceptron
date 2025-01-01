import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        
        # Initialize biases
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, inputs):
        # Input to hidden layer
        self.hidden_output = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        
        # Hidden to output layer
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)
        
        return self.output
    
    def backward(self, inputs, targets, learning_rate):
        # Compute output layer error
        output_error = targets - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)
        
        # Compute hidden layer error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.weights_input_hidden += learning_rate * np.dot(inputs.T, hidden_delta)
        self.bias_output += learning_rate * np.sum(output_delta, axis=0)
        self.bias_hidden += learning_rate * np.sum(hidden_delta, axis=0)
        
    def train(self, inputs, targets, learning_rate, epochs):
        for epoch in range(epochs):
            self.forward(inputs)
            self.backward(inputs, targets, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(targets - self.output))
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
                
    def predict(self, inputs):
        return self.forward(inputs)

# Load data from Excel file
def load_data(file_path):
    df = pd.read_excel(file_path)
    
    # Extract features and labels
    X = df.iloc[2:, 1:].values  # Exclude the first row and first column
    y = df.iloc[2:, 0].values   # Extract the first column as labels
    
    return X, y

# Example usage
file_path = "train/Train Dataset1.xlsx"  # Change file path accordingly
X, y = load_data(file_path)

# Normalize features if necessary
X_normalized = (X - X.mean()) / X.std()

# Initialize neural network
input_size = X_normalized.shape[1]
hidden_size = 4  # Adjust as needed
output_size = len(np.unique(y))  # Number of unique classes
learning_rate = 0.05
epochs = 5000

model = NeuralNetwork(input_size, hidden_size, output_size)

# Train the model
model.train(X_normalized, y, learning_rate, epochs)

# Test the model (if needed)
# predictions = model.predict(X_test_normalized)
# print("Predictions:")
# print(predictions)
