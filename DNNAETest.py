import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dummytrain = [[9, 30, 3, 184],
              [5, 44, 4, 146],
              [16, 41, 1, 91],
              [7, 22, 4, 209],
              [13, 34, 6, 132],
              [7, 30, 0, 55],
              [1, 19, 6, 56]]

xDummy = [[9, 30, 3],
          [5, 44, 4],
          [16, 41, 1],
          [7, 22, 4],
          [13, 34, 6],
          [7, 30, 0],
          [1, 19, 6]]
yDummy = [[184],
          [146],
          [91],
          [209],
          [132],
          [55],
          [56]]
# Normalize range from -1 to +1

class FFBB:
    def __init__(self, numPerLayer, epochs, eta):

        self.epochs = epochs
        self.eta = eta
        self.weights = []
        self.bias = []
        self.activity = []
        self.activation = []

        layers = len(numPerLayer)
        for layer in range(layers-1):
            # He-et-al Weight initialization
            self.weights.append(np.random.rand(numPerLayer[layer+1], numPerLayer[layer]) * np.sqrt(2/numPerLayer[layer]))
        for layer in range(1, layers):   # Generating bias except for input layer
            self.bias.append(np.random.rand(numPerLayer[layer]))

    def showWeights(self):
        print(self.weights)
        print(self.bias)

    def deltatanh(self, x):
        return 1.0 - np.tanh(x)**2

    def forward(self, input, y):
        if len(input) != self.weights[0].shape[1]:
            raise Exception('Invalid Input Size')
        output = 0
        for weight, bias in zip(self.weights, self.bias):
            activity = np.dot(input, weight.T) + bias
            self.activity.append(activity)
            input = np.tanh(activity)
            self.activation.append(input)
            output = input
        self.error = 0.5 * np.power(output-y, 2)
        self.deltaE = output - y
        return output

    def backpropagate(self):
        delta = self.deltaE * self.deltatanh(self.activity[-1])
        self.weights[-1] -= self.eta * self.activation[-1] * delta
        self.bias[-1] -= self.eta * delta
        for (index, array), z in zip(reversed(list(enumerate(self.activation[:-1]))), reversed(self.activity[:-1])):
            delta = np.dot(self.weights[index+1].T, delta) * self.deltatanh(z)
            self.weights[index] -= self.eta * np.dot(delta, array)
            self.bias[index] -= self.eta * delta

    def train(self, x, y):
        length = len(x)
        for index in range(self.epochs):
            avgError = 0
            for data, label in zip(x, y):
                # print(data, label)
                self.forward(data, label)
                avgError = avgError + self.error
                # print(avgError)
                self.backpropagate()
                self.activation = []
                self.activity = []
            print("iteration #{} Error: {}" .format(index+1, avgError/length))

    def predict(self, x):
        result = self.forward(x, 0)
        self.error = 0
        self.activation = []
        self.activity = []
        return result


df = pd.DataFrame([
    [9, 30, 3, 184],
    [3, 15, 1, 90],
    [2, 17, 3, 92],
    [6, 4, 2, 110]],

    columns=['Col A', 'Col B',
             'Col C', 'Col D'])
def normalize(data):
    for column in data.columns:
        data[column] = 2 * ((data[column] - data[column].min())/(data[column].max() - data[column].min())) - 1
    return data
