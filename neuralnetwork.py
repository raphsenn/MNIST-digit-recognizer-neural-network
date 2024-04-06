import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, input_neurons: int=784, hidden_neurons: int=10, output_neurons: int=10) -> None:
        """
        - 785 Input Neurons (for each pixel, MNIST 28x28px image)
        - 10 Hidden Neurons
        - 10 Output Neurons (for numbers 0 - 9)
        """ 
        self.w1 = np.zeros((input_neurons, hidden_neurons))     # 785 x 10
        self.b1 = np.zeros(hidden_neurons)                      # 10 x 1

        self.w2 = np.zeros((hidden_neurons, output_neurons))    # 10 x 10
        self.b2 = np.zeros(output_neurons)                      # 10 x 1

    def ReLU(self, x: np.array, derv: bool=False) -> np.array:
        if derv:
            return x > 0
        return np.maximum(0, x)

    def softmax(self, x: np.array) -> np.array:
        return np.exp(x) / sum(np.exp(x))

    def one_hot(self, y: np.array) -> np.array:
            one_hot_y = np.zeros((y.size, y.max() + 1))
            one_hot_y[np.arange(y.size), y] = 1
            return one_hot_y

    def train(self, X: np.array, y: np.array, epochs: int=5000, learning_rate: float=0.001, verbose: bool=False) -> None:
        for epoch in range(epochs):
            # Forward propagation.
            Z1 = np.dot(X, self.w1) + self.b1                               # m x 10
            A1 = self.ReLU(Z1)                                              # m x 10
            Z2 = np.dot(A1, self.w2) + self.b2                              # m x 10
            A2 = self.softmax(Z2) # Predictions.                            # m x 10
            one_hot_y = self.one_hot(y)                                     # m x 10

            # Calculate loss. 
            # loss = np.mean((A2 - one_hot_y.reshape(-1, 1)) ** 2)
            # loss = np.mean((A2 - one_hot_y) ** 2)

            # Backpropagation.
            m = one_hot_y.size
            dZ2 = A2 - one_hot_y # Output error                             # m x 10
            dW2 = 1/m * np.dot(dZ2.T, A1)                                   # 10 x 10
            db2 = 1/m * np.sum(dZ2)                                         # 1 x 1
            
            dZ1 = np.dot(dZ2, self.w2) * self.ReLU(Z1, derv=True)         # m x 10
            dW1 = 1/m * np.dot(X.T, dZ1)                                    # 785 x 10
            db1 = 1/m * np.sum(dZ1)                                         # 1 x 1

            self.w2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.w1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

            # Calculate loss. 
            loss = np.mean((A2 - one_hot_y) ** 2)
            
            # Backpropagation. 
            #output_error = 2 * (A2 - one_hot_y)
            #hidden_error = np.dot(output_error, self.w2.T) * A1 * (1 - A1)
            #self.w2 -= learning_rate * np.dot(A1.T, output_error) / len(y)
            #self.b2 -= learning_rate * np.sum(output_error) / len(y)
            #self.w1 -= learning_rate * np.dot(X.T, hidden_error) / len(y)
            #self.b1 -= learning_rate * np.sum(hidden_error) / len(y)

            if verbose:
                if epoch % 1 == 0:
                    # print(f"Epoch {epoch}")
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")
                    #print(f"Z1: {Z1.shape}")
                    #print(f"A1: {A1.shape}")
                    #print(f"Z2: {Z2.shape}")
                    # print(f"A2: {A1.shape}")
                    #print(f"y: {y.shape}")
                    #print(f"dZ2: {dZ2.shape}")
                    print(f"dZ1: {dZ1.shape}")
                    # print(f"db2: {db2.shape}")
                    # print(f"dW2: {dW2.shape}")
                    # print(f"m: {m}")

    def predict(self, X: np.array) -> np.array:
        # Forward propagation.
        Z1 = np.dot(X, self.w1) + self.b1
        A1 = self.ReLU(Z1)
        Z2 = np.dot(A1, self.w2) + self.b2
        A2 = self.softmax(Z2)
        output = np.argmax(A2, 1)
        return output

    def evaluate(self, X, y):
        # Calculate predictions. 
        predictions = self.predict(X)
        print(y)
        print(predictions)

        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(len(y)):
            if predictions[i] == 1 and y[i] == 1:
                TP += 1
            elif predictions[i] == 0 and y[i] == 0:
                TN += 1
            elif predictions[i] == 1 and y[i] == 0:
                FP += 1
            else:
                FN += 1
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN != 0 else 0
        return precision, recall, f1, accuracy
    
def read_data(file: str) -> tuple[np.array, np.array]: 
    data = pd.read_csv(file)
    data = np.array(data).T
    X, y  = data[1:].T / 255, data[0]
    return X, y


if __name__ == '__main__':
    X_train, y_train = read_data('train.csv')
    # X_test, y_test = read_data('test.csv')
    
    # Create a NeuralNet. 
    nn = NeuralNetwork()
    
    # Train the NeuralNet.
    nn.train(X_train, y_train, 10, 0.1, True)
    
    # Evaluate on testing data.
    precision, recall, f1, accuracy = nn.evaluate(X_train, y_train)
    print(f"precision = {round(precision * 100, 2)}%")
    print(f"recall = {round(recall * 100, 2)}%")
    print(f"F1 = {round(f1 * 100, 2)}%")
    print(f"accuracy = {round(accuracy * 100, 2)}%")
    print()