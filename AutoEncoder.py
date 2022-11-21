import numpy as np
import pandas as pd
from DNNAETest import FFBB
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import DNNReg as DNN
import LinearClassification as LC

def normalize(data):
    for column in data.columns:
        data[column] = 2 * ((data[column] - data[column].min())/(data[column].max() - data[column].min())) - 1
    return data

xDummy = [[9, 30, 3],
          [5, 44, 4],
          [16, 41, 1],
          [7, 22, 4],
          [13, 34, 6],
          [7, 30, 0],
          [1, 19, 6],
          [2, 19,1],
          [7, 30, 0],
          [1, 19, 6],
          [2, 19, 1]]

xDummy = pd.DataFrame(xDummy)
xDummy = normalize(xDummy)
xNP = np.array(xDummy)

class autoencoder:
    def __init__(self, numPerLayer, epochs, eta):

        self.epochs = epochs
        self.eta = eta
        self.weights = []
        self.activity = []
        self.activation = []

        layers = len(numPerLayer)
        for layer in range(layers - 1):
            # He-et-al Weight initialization
            self.weights.append(np.random.rand(numPerLayer[layer + 1], numPerLayer[layer]) * np.sqrt(2 / numPerLayer[layer]))

    def showWeights(self):
        print(self.weights)

    def encoder(self, data):
        if len(data) != self.weights[0].shape[1]:
            raise Exception('Invalid Input Size')
        activation = []
        for weight in self.weights[0]:
            activity = np.dot(data, weight.T)
            self.activity.append(activity)
            input = np.tanh(activity)
            self.activation.append(input)
            activation.append(input)
        print('These is the activations\n', activation)
        return activation

    def decoder(self, input):
        decoded = np.dot(input, self.weights[1].T)
        return decoded

    def train(self, data):
        length = len(data)
        weights = []
        meanError = []
        for epoch in range(self.epochs):
            avgError = 0
            for x in data:
                # weights.append(self.weights)
                print('Start - This is X\n', x)
                results = self.encoder(x)
                output = self.decoder(results)
                print("This is the decoding (output) results\n", output)
                print('This is the X actual\n', x)
                self.error = 0.5 * np.power(output - x, 2)
                avgError = avgError + self.error
                self.deltaE = output - x
                print('This is the error\n', self.error)
                print('This is the delta error\n', self.deltaE)
                self.backprop()
                self.activation = []
                self.activity = []
            meanError.append(np.mean(avgError))
            weights.append(self.weights)
            # print("iteration #{} Error: {}".format(epoch + 1, avgError / length))
            # print('this is the mean error: {}'.format(np.mean(avgError)))
        return weights, meanError

    def deltatanh(self, x):
        return 1.0 - np.tanh(x)**2

    def backprop(self):
        print('Initial weights\n', self.weights)
        delta = self.deltaE * self.deltatanh(self.activity[-1])
        updateValue = self.eta * self.activation[-1] * delta
        idMatrix = np.ones(self.weights[-1].shape)
        UpdateMatrix = np.multiply(idMatrix.T, updateValue)
        self.weights[-1] = np.subtract(self.weights[-1], UpdateMatrix.T)
        # print('All weights\n',self.weights, np.shape(self.weights))
        # print('test', self.weights[0][1])
        # print('All activations\n', self.activation)
        for (index, array), z in zip(reversed(list(enumerate(self.activation[:-1]))), reversed(self.activity[:-1])):
            # print('index', index)
            # print('array\n', array)
            # print('z\n', z)
            length = len(self.weights)
            # print(length)
            for index in range(length):
                # print('Look at this test')
                # print(index)
                # print(self.weights[index-1])
                # print('Current weight\n', self.weights[index-1])
                # print('Delta\n', delta)
                delta = np.dot(self.weights[index-1].T, delta * self.deltatanh(z))
                # print('Calc Delta\n', delta)
                updateValues = self.eta * np.dot(delta, array)
                # print('UpdatedValues\n', updateValues)

                idMatrix = np.ones(self.weights[index].shape)
                # print('idMatrix\n', idMatrix)
                updateMatrix = np.multiply(idMatrix.T, updateValues)
                # print('updateMatrix\n',updateMatrix)
                self.weights[index] = np.subtract(self.weights[index], updateMatrix.T)
            # Testing
            # delta = np.dot(self.weights[index + 1].T, delta * self.deltatanh(z))
            # updateValues = self.eta * np.dot(delta, array)
            # idMatrix = np.ones(self.weights[index].shape)
            # updateMatrix = np.multiply(idMatrix.T, updateValues)
            # self.weights[index] = np.subtract(self.weights[index], updateMatrix.T)
        print('Updated Weights\n', self.weights)


class AENN:
    def __init__(self, data, k, epochs, eta, regStrength, momentum):
        """
        :param data: whole data??
        """
        print('This is how many Features X has\n', data.shape[1]-1)
        self.AELayers = [data.shape[1]-1, data.shape[1]-2, data.shape[1]-1]
        print('This is the Autoencoding Layers\n',self.AELayers)
        self.DNNLayers = [data.shape[1]-2, data.shape[1]-1, data.shape[1]-2, 1]
        print('This is the DNN Layers\n', self.DNNLayers)

        self.k = k
        self.epochs = epochs
        self.eta = eta
        self.regStrength = regStrength
        self.momentum = momentum


    def trainAE(self, xTrain):
        ae = autoencoder(self.AELayers, self.epochs, self.eta)
        weights, meanError = ae.train(xTrain)
        minValue = min(meanError)
        minIndex = meanError.index(minValue)
        bestWeightsAE = weights[minIndex][0]
        return bestWeightsAE

    def encode(self, x, bestWeights):
        # print(np.shape(x))
        # print(bestWeights[0])
        encoded = np.dot(x, bestWeights.T)
        # print(np.shape(encoded))
        return encoded

    # Was going to use this fit originally but decided to process it through the
    # cross validation from the DNNreg.py
    def fit(self, x, y, epochs, eta):
        print('Initializing Traditional DNN')
        traditional = FFBB(self.DNNLayers, epochs, eta)
        bestWeights = self.trainAE(x)
        print('Best Weights After Training Autoencoder', bestWeights)
        encodedX = self.encode(x, bestWeights)
        print('X Encoded from Trained Autoencoders\n', encodedX)
        print('Now attached Attaching To Other Network')
        traditional.train(encodedX, y)
        trainingResults = []
        traditional.showWeights()
        for x in encodedX:
            prediction = traditional.predict(x)
            trainingResults.extend(prediction)
        self.scatterplot(y, trainingResults)

    def fitREG(self, data):
        x = data[:,:-1]
        y = data[:, -1]
        print('Initializing Traditional DNN\n')
        bestWeights = self.trainAE(x)
        print('Best Weights After Training Autoencoder\n', bestWeights)
        encodedX = self.encode(x, bestWeights)
        print('X Originally\n', x)
        print('X Encoded from Trained Autoencoders\n', encodedX)
        print('Now attached Attaching To Other Network\n')
        encodedDF = pd.DataFrame(encodedX)
        encodedDF['Y'] = y
        print('Dataframe\n', encodedDF)
        encodedNP = encodedDF.to_numpy()
        print('Numpy\n', encodedNP)
        DNN.CrossValidation(encodedNP, self.k ,self.DNNLayers, self.epochs, self.eta, self.regStrength, self.momentum)

    def fitCLASS(self, data):
        x = data[:,:-1]
        y = data[:, -1]
        print('Initializing Traditional DNN\n')
        bestWeights = self.trainAE(x)
        print('Best Weights After Training Autoencoder\n', bestWeights)
        encodedX = self.encode(x, bestWeights)
        print('X Originally\n', x)
        print('X Encoded from Trained Autoencoders\n', encodedX)
        print('Now attached Attaching To Other Network\n')
        encodedDF = pd.DataFrame(encodedX)
        encodedDF['Y'] = y
        print('Dataframe\n', encodedDF)
        encodedNP =  encodedDF.to_numpy()
        print('Numpy\n', encodedNP)
        # .CrossValidation(encodedNP, self.k ,self.DNNLayers, self.epochs, self.eta, self.regStrength, self.momentum)
        LC.CrossValidation(data, self.k, self.epochs, self.eta, 3, self.regStrength, self.momentum)

    def scatterplot(self, yActual, yPredicted):
        df = pd.DataFrame({'Actual':yActual, 'Predicted': yPredicted})
        sns.set_style('ticks')
        sns.regplot(x='Actual', y='Predicted', data=df, fit_reg =True,
                    scatter_kws={'color': 'darkred', 'alpha': 0.3, 's': 100})
        plt.show()
        print('Look', df)
        return df




# DUMMY TESTING
df = pd.DataFrame([
    [9, 30, 3, 4, 184],
    [3, 15, 1, 2,  90],
    [2, 17, 3, 3,  92],
    [6, 4, 2, 3, 110],
    [3, 15, 1, 2,  90],
    [2, 17, 3, 3,  92],
    [6, 4, 2, 3, 110]],

    columns=['Col A', 'Col B',
             'Col C', 'Col D', 'Y' ])

# dftest = normalize(df)
# dfnp = df.to_numpy()
# x = dfnp[:, :-1]
# y = dfnp[:, -1]
# print('X-Values\n', x)
# print('Y-Values\n', y)
# test = autoencoder([3,2,3], 100, 0.02)
# test.train(xNP)

# test = AENN(dfnp,k=5, epochs = 100, eta=0.03, regStrength=0.001, momentum=0.05)
# print('Start Here')
# test.fitREG(dfnp)
#
#
