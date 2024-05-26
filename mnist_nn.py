import numpy as np
from mnist import MNIST
import time

# Load the MNIST dataset
mndata = MNIST('samples')
x_train, y_train = mndata.load_training()  # Loads the training images and labels.
x_test, y_test = mndata.load_testing()  # Loads the test images and labels.

# Convert to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Normalize the images to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to one-hot encoding
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

y_train = one_hot_encode(y_train, 10)
y_test = one_hot_encode(y_test, 10)

# Initialize weights
def initialize_weights(input_size, hidden_size, output_size):
    wh = np.random.randn(input_size, hidden_size) * 0.01
    wo = np.random.randn(hidden_size, output_size) * 0.01
    return wh, wo

# Sigmoid activation function with gradient clipping
def sigmoid(z):
    # Clip the values of z to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Derivative of the sigmoid function
def sigmoid_derivative(z):
    return z * (1 - z)

# Forward propagation
def forward_propagation(x, wh, wo):
    hidden_layer_input = np.dot(x, wh)
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, wo)
    output_layer_output = sigmoid(output_layer_input)
    
    return hidden_layer_output, output_layer_output

# Backward propagation
def backward_propagation(x, y, hidden_layer_output, output_layer_output, wh, wo, learning_rate):
    error_output_layer = y - output_layer_output
    d_output = error_output_layer * sigmoid_derivative(output_layer_output)
    
    error_hidden_layer = d_output.dot(wo.T)
    d_hidden = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    wo += hidden_layer_output.T.dot(d_output) * learning_rate
    wh += x.T.dot(d_hidden) * learning_rate
    
    return wh, wo

# Training the neural network
def train(x_train, y_train, epochs, learning_rate, timeout=None):
    input_size = x_train.shape[1]
    hidden_size = 128  # Number of neurons in the hidden layer
    output_size = 10  # Number of output classes

    wh, wo = initialize_weights(input_size, hidden_size, output_size)
    start_time = time.time()
    
    for epoch in range(epochs):
        hidden_layer_output, output_layer_output = forward_propagation(x_train, wh, wo)
        wh, wo = backward_propagation(x_train, y_train, hidden_layer_output, output_layer_output, wh, wo, learning_rate)
        
        if (epoch + 1) % 100 == 0:
            loss = np.mean(np.square(y_train - output_layer_output))
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

        if timeout and time.time() - start_time > timeout:
            print(f"Training interrupted after {timeout} seconds.")
            break

    return wh, wo

# Prediction function
def predict(x, wh, wo):
    _, output_layer_output = forward_propagation(x, wh, wo)
    return np.argmax(output_layer_output, axis=1)

# Train the neural network with a timeout of 300 seconds (5 minutes)
epochs = 1000
learning_rate = 0.01
timeout = 300  # 5 minutes
wh, wo = train(x_train, y_train, epochs, learning_rate, timeout=timeout)

# Evaluate the neural network
y_pred = predict(x_test, wh, wo)
accuracy = np.mean(np.argmax(y_test, axis=1) == y_pred)
print(f"Test accuracy: {accuracy:.4f}")
