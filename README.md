# Neural Network [Numpy]

This repository provides an implementation of a simple **feedforward neural network** trained to solve the XOR problem. The main goal is to demonstrate how to build and train a basic neural network using **NumPy** and visualize training progress with the **colorful** library.

## Overview

The project implements a simple neural network with a single hidden layer to predict the XOR function's output for a pair of binary inputs. The network includes customizable training parameters like the number of epochs and learning rate, making it easy to understand how these factors influence training.

### Features
- **NumPy-based Implementation**: Fully implemented with NumPy, providing flexibility and transparency into each computational step.
- **XOR Problem Solver**: Designed to solve the XOR problem, showcasing the basic workings of neural networks.
- **Colorful Output**: Uses the `colorful` library to make output more visually engaging, highlighting the training progress.

### Architecture

The neural network has the following architecture:

1. **Input Layer**: Takes two binary inputs.
2. **Hidden Layer**: Consists of four neurons, responsible for learning intermediate representations.
3. **Output Layer**: A single neuron providing the final prediction.

The activation function used is the **sigmoid** function for both the hidden and output layers, which ensures the output values are between 0 and 1.

## Usage

The script can be used to train the neural network on the XOR dataset. The general workflow includes:

1. **Define the Dataset**: The XOR truth table is used as the dataset.
2. **Initialize Weights and Biases**: Random initialization is performed for weights and biases to start the training process.
3. **Train the Model**: The `train()` function is used to train the model for a specified number of epochs.
4. **Make Predictions**: Use the `predict()` function to see how well the model has learned the XOR logic.

### Example Workflow

```python
# Training data (XOR truth table)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Training parameters
epochs = 60000
learning_rate = 0.1

# Train the model
train(X, y, epochs, learning_rate)

# Make predictions
print(cf.blue(predict(np.array([[0, 0]]))))
print(cf.blue(predict(np.array([[0, 1]]))))
print(cf.blue(predict(np.array([[1, 0]]))))
print(cf.blue(predict(np.array([[1, 1]]))))
```

### Requirements

- **NumPy**: Required for numerical computations.
- **Colorful**: Provides colorful console output to enhance the user experience.

To install the required dependencies, run:

```sh
pip install numpy colorful
```

## How it Works

The model is trained using the **backpropagation** algorithm. During training:

1. **Forward Propagation**: Calculates the output of the hidden layer and the final prediction.
2. **Error Calculation**: The difference between the predicted and actual output is calculated.
3. **Backpropagation**: Gradients are computed to update weights and biases using the **sigmoid derivative**.

The training progress is displayed every 1000 epochs, showing the loss value to understand how the model improves over time.
