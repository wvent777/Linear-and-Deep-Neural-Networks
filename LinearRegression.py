import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def normalize(data):    # normalizes data between -1 and 1
    for column in data.columns:
        data[column] = 2 * ((data[column] - data[column].min())/(data[column].max() - data[column].min())) - 1
    return data


# dummy2train = [[9, 30, 3, 184],
#               [5, 44, 4, 146],
#               [16, 41, 1, 91],
#               [7, 22, 4, 209],
#               [13, 34, 6, 132],
#               [7, 30, 0, 55],
#               [1, 19, 6, 56]]
#
# dummy2train = pd.DataFrame(dummy2train, columns=['X1', 'X2', 'X3', 'Y'])
# normalize(dummy2train)
# dummy2train = dummy2train.to_numpy()
# print(dummy2train)

class LinearRegression:
    def __init__(self, epochs, eta, batchSize, regStrength, momentum):
        self.epochs = epochs
        self.eta = eta
        self.bias = 0
        self.batchSize = batchSize
        self.regStrength = regStrength
        self.momentum = momentum
        self.velocity = None
        self.weight = None

    def activity(self, X):
        return np.dot(X, self.weight[:-1]) + self.bias * self.weight[-1]

    def mseLoss(self, X, Y):
        numSamples = X.shape[0]
        activity = self.activity(X)
        activity = np.tanh(activity)
        regLoss = (1/2) * self.regStrength * np.sum(self.weight * self.weight)
        totalLoss = (np.sum(activity)/numSamples) + regLoss

        gradients = ((-1 / numSamples) * np.dot(X.T, (Y - activity))) + (self.regStrength * self.weight[:-1])
        biasGradient = ((-1 / numSamples) * np.mean((Y-activity), axis=0)) + (self.regStrength * self.weight[-1])
        gradients = np.append(gradients,biasGradient)
        # print(gradients)
        return totalLoss, gradients

    def SGD(self, X, Y):
        losses = []
        for index in range(0, X.shape[0], self.batchSize):
            xBatch = X[index:index + self.batchSize]
            yBatch = Y[index:index + self.batchSize]

            loss, deltaW = self.mseLoss(xBatch, yBatch)
            self.velocity = (self.momentum * self.velocity) + (self.eta * deltaW)
            self.weight -= self.velocity
            losses.append(loss)
        return np.sum(losses)/len(losses)

    def predict(self, X):
        return self.activity(X)

    def mse(self, X, Y):
        yPred = self.predict(X)
        return (np.square(Y - yPred)).mean(axis=None)

    def train(self, xTrain, yTrain, xTest, yTest):
        dimensionality = xTrain.shape[1]

        self.weight = np.random.randn(1 + dimensionality) * 0.01
        self.velocity = np.zeros(self.weight.shape)

        trainLosses = []
        testLosses = []
        trainMSE = []
        testMSE = []
        trainPredicted = []
        testPredicted = []
        for epoch in range(self.epochs):
            trainLoss = self.SGD(xTrain, yTrain)
            testLoss, deltaW = self.mseLoss(xTest, yTest)

            trainPredicted.append(self.predict(xTrain))
            testPredicted.append(self.predict(xTest))

            trainMSE.append(self.mse(xTrain, yTrain))
            testMSE.append(self.mse(xTest, yTest))

            trainLosses.append(trainLoss)
            testLosses.append(testLoss)
            print("{:d}\t->\tTrainL : {:.7f}\t|\tTestL : {:.7f}\t|\tTrainMSE : {:.7f}\t|\tTestMSE: {:.7f}"
                  .format(epoch, trainLoss, testLoss, trainMSE[-1], testMSE[-1]))
        return trainLosses, testLosses, trainMSE, testMSE, trainPredicted[-1], testPredicted[-1]


def scatterplot(trainActual, trainPredicted, testActual, testPredicted):
    traindf = pd.DataFrame({'Train Actual':trainActual, 'Train Predicted': trainPredicted})
    testdf = pd.DataFrame({'Test Actual': testActual, 'Test Predicted': testPredicted})

    plt.figure(figsize=(10,7))
    plt.suptitle('Actual vs. Predicted', fontsize= 24)
    sns.set_style('whitegrid')
    plt.subplot(1,2,1)
    sns.regplot(x='Train Predicted', y='Train Actual', data=traindf, fit_reg =True, ci = 95,
                scatter_kws={'color': 'purple', 'alpha': 0.3, 's': 80},
                line_kws={'color':'#CCCC00', 'alpha': 0.3})

    plt.subplot(1,2,2)
    sns.regplot(x='Test Predicted', y='Test Actual', data=testdf, fit_reg=True, ci = 95,
                scatter_kws={'color': 'green', 'alpha': 0.3, 's': 80},
                line_kws = {'color': '#CCCC00', 'alpha': 0.3})
    plt.show()

def plotGraph(trainLosses, testLosses, trainMSE, testMSE):
    length = len(trainLosses)
    fig, (ax1, ax2) = plt.subplots(2, length, sharex=True, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 1]})
    fig.supxlabel('Number of Epochs', fontsize = 15)
    plt.suptitle('Total Loss & Mean Square Error vs. Epochs for K(i) Fold', fontsize =18)
    for index in range(length):
        ax1[index].plot(trainLosses[index], label="Train Total Loss")
        ax1[index].plot(testLosses[index], label="Test Total Loss")

        ax2[index].plot(trainMSE[index], label="Train MSE", color='red')
        ax2[index].plot(testMSE[index], label="Test MSE", color='purple')
        ax1[index].set_ylabel('Total Loss', fontsize= 9)
        ax2[index].set_ylabel('Mean Square Error', fontsize = 9)
        handles, labels = [(a + b) for a, b in zip(ax1[index].get_legend_handles_labels(), ax2[index].get_legend_handles_labels())]
        fig.legend(handles, labels, loc='center right')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.85,
                        wspace=0.5,
                        hspace=0.2)
    plt.show()

    avgtrainMSE = np.mean(trainMSE, axis=0)
    avgtestMSE = np.mean(testMSE, axis=0)

    mintrainMSE = min(avgtrainMSE)
    mintestMSE = min(avgtestMSE)

    plt.plot(avgtrainMSE, label="Avg. Train MSE", color = 'red')
    plt.plot(avgtestMSE, label="Avg. Test MSE", color = 'magenta')
    plt.legend(loc='best')
    plt.title("Cross Validation Avg. Mean Square Error vs. Epochs")
    plt.xlabel("Number of Epochs", fontsize=7)
    plt.ylabel("Mean Square", fontsize=8)
    txt = f'The Minimum Average MSE is Training: {mintrainMSE}, Testing: {mintestMSE}'
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=7)
    plt.show()


def LinearRegressionAlgo(trainData, testData, epochs, eta, batchSize, regStrength, momentum):
    xTrain = trainData[:, :-1]
    yTrain = trainData[:, -1]
    # yTrain = yTrain.reshape((-1, 1))
    xTest = testData[:, :-1]
    yActual = testData[:, -1]
    # yActual = yActual.reshape((-1, 1))
    nn = LinearRegression(epochs=epochs, eta=eta, batchSize=batchSize, regStrength=regStrength, momentum=momentum)
    trainlosses, testlosses, trainmse, testmse, trainpred, testpred = nn.train(xTrain, yTrain, xTest, yActual)
    return trainlosses, testlosses, trainmse, testmse, trainpred, testpred, yTrain, yActual


def CrossValidation(data, k, epochs, eta, batchSize, regStrength, momentum):
    np.random.shuffle(data)
    folds = np.array_split(data, k)
    trainlosses = []
    testlosses = []
    trainmse = []
    testmse = []

    trainPreds = []
    testPreds = []
    yTrainList = []
    yActualList = []
    for i in range(k):
        testData = folds[i][:, :]
        newFolds = np.row_stack(np.delete(folds, i, 0))
        trainData = newFolds[:, :]
        print(f'Fold {i + 1}')
        print(f'Testing set size : {testData.shape}')
        print(f'Training set size : {trainData.shape}\n')
        foldtrainloss, foldtestloss, foldtrainacc, foldtestacc, foldtrainPred, foldtestPred, foldyTrain, foldyActual = LinearRegressionAlgo(trainData,
                                                    testData, epochs, eta, batchSize, regStrength, momentum)
        trainlosses.append(foldtrainloss)
        testlosses.append(foldtestloss)

        trainmse.append(foldtrainacc)
        testmse.append(foldtestacc)

        trainPreds.extend(foldtrainPred)
        testPreds.extend(foldtestPred)

        yTrainList.extend(foldyTrain)
        yActualList.extend(foldyActual)

    plotGraph(trainlosses, testlosses, trainmse, testmse)
    scatterplot(yTrainList, trainPreds, yActualList, testPreds)
    # yTrainList = np.concatenate(yTrainList).ravel().tolist()
    # yActualList = np.concatenate(yActualList).ravel().tolist()
    return





# dummy2np = np.array(dummy2train)
# dummy2XTrain = dummy2np[:-1, :-1]
# dummy2XTest = dummy2np[-1, :-1]
#
# dummy2YTrain = dummy2np[:-1, -1]
# dummy2YTest = dummy2np[-1, -1]
#
#
# CrossValidation(dummy2np, 5, 300, 0.02, 3, 0.001, 0.005)

