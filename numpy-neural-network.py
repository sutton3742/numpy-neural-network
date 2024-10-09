import numpy as np
import colorful as cf

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize seed for reproducibility
np.random.seed(1)

# Number of inputs, neurons in hidden layer, and outputs
input_size = 2
hidden_size = 4
output_size = 1

# Initialize weights and biases
weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_hidden = np.random.rand(1, hidden_size)
bias_output = np.random.rand(1, output_size)

def train(X, y, epochs, learning_rate):
    global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output
    
    for epoch in range(epochs):
        # Forward propagation
        hidden_layer_activation = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_activation)
        
        output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        predicted_output = sigmoid(output_layer_activation)
        
        # Compute error
        error = y - predicted_output
        
        # Backpropagation
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
        
        # Update weights and biases
        weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
        weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
        bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        
        if (epoch + 1) % 1000 == 0:
            loss = np.mean(np.square(error))
            print(cf.red(f'Epoch {epoch + 1}, Loss: {loss}'))

# Training data (simple example)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Training parameters
epochs = 60000
learning_rate = 0.1

train(X, y, epochs, learning_rate)

def predict(X):
    hidden_layer_activation = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_activation)
    return predicted_output

# Make predictions
print(cf.blue(predict(np.array([[0, 0]]))))
print(cf.blue(predict(np.array([[0, 1]]))))
print(cf.blue(predict(np.array([[1, 0]]))))
print(cf.blue(predict(np.array([[1, 1]]))))