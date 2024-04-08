import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, input_neurons: int=784, hidden_neurons: int=10, output_neurons: int=10) -> None:
        """
        - 785 Input Neurons (for each pixel, MNIST 28x28px image)
        - 10 Hidden Neurons
        - 10 Output Neurons (for numbers 0 - 9)
        """ 
        self.w1 = np.random.rand(hidden_neurons, input_neurons) - 0.5
        self.b1 = np.random.rand(hidden_neurons, 1) - 0.5
        self.w2 = np.random.rand(hidden_neurons, output_neurons) - 0.5
        self.b2 = np.random.rand(output_neurons, 1) - 0.5

    def ReLU(self, x: np.array, derv: bool=False) -> np.array:
        if derv:
            return x > 0
        return np.maximum(x, 0)

    def softmax(self, x: np.array) -> np.array:
        return np.exp(x) / sum(np.exp(x))

    def one_hot(self, Y: np.array) -> np.array:
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def train(self, X: np.array, y: np.array, epochs: int=5000, learning_rate: float=0.1, verbose: bool=False) -> None:
        for epoch in range(epochs):
            # Forward propagation.
            Z1 = np.dot(self.w1, X) + self.b1
            A1 = self.ReLU(Z1)
            Z2 = np.dot(self.w2, A1) + self.b2
            A2 = self.softmax(Z2)
            
            one_hot_y = self.one_hot(y)
            m, _ = X.shape

            # Backpropagation.
            dZ2 = A2 - one_hot_y # error
            dW2 = 1/m * np.dot(dZ2, A1.T)
            db2 = 1/m * np.sum(dZ2)

            dZ1 = np.dot(self.w2.T, dZ2) * self.ReLU(Z1, derv=True)
            dW1 = 1/m * np.dot(dZ1, X.T)
            db1 = 1/m * np.sum(dZ1)

            # Update weights and bias.
            self.w1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.w2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            # Calculate loss. 
            loss = np.mean(A2 - one_hot_y)

            if verbose:
                if epoch % 10 == 0:
                    print()
                    predictions = self.get_predictions(A2) 
                    print(predictions, y)
                    print(f"Epoch {epoch}, Loss: {self.get_accuracy(predictions, y)}")

    def get_accuracy(self, predictions, y):
        return np.sum(predictions == y) / y.size

    def get_predictions(self, A2):
        return np.argmax(A2, 0)
    
    def predict(self, X: np.array) -> np.array:
        # Forward propagation.
        Z1 = np.dot(self.w1, X) + self.b1
        A1 = self.ReLU(Z1)
        Z2 = np.dot(self.w2, A1) + self.b2
        A2 = self.softmax(Z2)
        # Output.
        output = np.argmax(A2, 0)
        return output


def read_data(file: str) -> tuple[np.array, np.array]: 
    data = pd.read_csv(file)
    data = np.array(data).T
    X, y  = data[1:] / 255, data[0]
    return X, y


if __name__ == '__main__':
    X_train, y_train = read_data('train.csv')
    
    # Create a NeuralNet. 
    nn = NeuralNetwork(784, 10, 10)
    
    # Train the NeuralNet.
    nn.train(X_train, y_train, 500, 0.001, True)