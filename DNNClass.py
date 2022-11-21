import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dummyTotalData = [[0.234,0.422,0.543,1],
[0.562,0.75,0.35,0],
[0.7513,0.21,0.012,1],
[0.943,0.4,0.75,0],
[0.7513, 0.21, 0.012,1],
[0.943, 0.4, 0.75,0],
[0.334,0.222,0.643,1],
[0.462,0.65,0.25,1],
[0.513,0.11,0.08,0],
[0.943,0.4,0.75,0]]

xTrain = [[0.234,0.422,0.543],
[0.562,0.75,0.35],
[0.7513,0.21,0.012],
[0.943,0.4,0.75],
[0.7513, 0.21, 0.012],
[0.943, 0.4, 0.75]]

yTrain = [[1],
[0],
[1],
[0],
[1],
[0]]

xTest = [[0.334,0.222,0.643],
[0.462,0.65,0.25],
[0.513,0.11,0.08],
[0.943,0.4,0.75]]

yTest = [[1],
[1],
[0],
[0]]

# Network Architecture -> input(Xn) -> (H1(Xn)) -> (H2(H1(Xn)) - > Output
# All Nodes must be same length

class DNNClass():
    def __init__(self, NNSize, epochs, eta, batchSize, regStrength, momentum, activation='sigmoid', optimizer='SGD'):
        """
        :param NNSize: NNSize of the nodes [input, Hidden1, Hidden2, Output]
        """
        self.NNSize = NNSize
        self.epochs = epochs
        self.eta = eta
        self.batchSize = batchSize
        self.regStrength = regStrength
        self.momentum = momentum
        self.optimizer = optimizer

        # Choose activation function
        #   Originally going to have regression in there too
        #   Later Moved to separate
        if activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            print('Wrong Activation Input')

        # Save all weights
        self.params = self.initialize()
        # Save all intermediate values, i.e. activations, activity
        self.cache = {}

    def oneHotEncoding(self, y, numOfClasses):
        y = np.asarray(y, dtype='int32')
        if len(y) > 1:
            y = y.reshape(-1)
        if not numOfClasses:
            numOfClasses = np.max(y) + 1
        yMatrix = np.zeros((len(y), numOfClasses))
        yMatrix[np.arange(len(y)), y] = 1
        return yMatrix

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    # Soft Max 2 Doesn't WorK
    def softmax2(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def softmax(self, x):
        x -=np.max(x)
        probability = (np.exp(x).T / np.sum(np.exp(x), axis=1)).T
        return probability

    def initialize(self):
        # number of nodes in each layer
        inputLayer = self.NNSize[0]
        H1 = self.NNSize[1]
        H2 = self.NNSize[2]
        outputLayer = self.NNSize[3]
        # He-et-al Initialization
        params = {
            "W1": np.random.randn(H1, inputLayer) * np.sqrt(2 / inputLayer),
            "b1": np.zeros((H1, 1)) * np.sqrt(1 / inputLayer),
            "W2": np.random.randn(H2, H1) * np.sqrt(2 / H1),
            "b2": np.zeros((H2, 1)) * np.sqrt(1 / H1),
            "W3": np.random.randn(outputLayer, H2) * np.sqrt(2 / H2),
            "b3": np.zeros((outputLayer, 1)) * np.sqrt(1 / H2)
        }
        return params

    def initMomentum(self):
        momentums = {
            "W1": np.zeros(self.params["W1"].shape),
            "b1": np.zeros(self.params["b1"].shape),
            "W2": np.zeros(self.params["W2"].shape),
            "b2": np.zeros(self.params["b2"].shape),
            "W3": np.zeros(self.params["W3"].shape),
            "b3": np.zeros(self.params["b3"].shape),
        }
        return momentums

    def forward(self, x):
        print('x\n', x)
        self.cache['X'] = x
        # Hidden 1
        print('Weights 1\n', self.params['W1'])
        self.cache['Z1'] = np.dot(self.params['W1'], self.cache['X'].T) + self.params['b1']
        self.cache['A1'] = self.activation(self.cache['Z1'])
        print('Hidden 1 Activation\n', self.cache['A1'])
        # Hidden2
        print('Weights 2\n', self.params['W2'])
        self.cache['Z2'] = np.dot(self.params['W2'], self.cache['A1']) + self.params['b2']
        self.cache['A2'] = self.activation(self.cache["Z2"])
        print('Hidden 2 Activation\n', self.cache['A2'])
        # Output
        self.cache['Z3'] = np.dot(self.params['W3'], self.cache['A1']) + self.params['b3']
        self.cache['A3'] = self.softmax(self.cache['Z3'])
        print('output\n', self.cache['A3'])
        return self.cache['A3']

    def backProp(self, y, output):
        # ERRORS CAN"T GET IT TO WORK RIGHT
        # print('y' , y , np.shape(y))
        # numSamples = y.shape[0]
        # print(y)
        # print('num', numSamples)
        # loss = -np.log(np.max(output)) * y
        # print('loss', np.shape(loss), loss)

        # regLoss = (1/2) * self.regStrength* np.sum(self.params['W3']*self.params['W3'])
        # totalLoss = (np.sum(loss, axis=0, keepdims=True)/numSamples)

        dZ3 = output - y.T


        dW3 = (1 / self.batchSize) * np.dot(dZ3, self.cache['A2'].T)
        db3 = (1 / self.batchSize) * np.sum(dZ3, axis=1, keepdims=True)

        h2Loss = np.dot(self.params['W3'], dW3.T)
        dA2 = np.dot(h2Loss, dZ3)
        dZ2 = np.dot(dA2, (self.activation(self.cache['Z2'], derivative=True)).T)

        dW2 = (1/ self.batchSize) * np.dot(dZ2, np.dot(self.cache['A1'],dZ3.T))
        db2 = (1 / self.batchSize) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.params['W2'].T, dZ2.T)
        dZ1 = np.dot(dA1.T, self.activation(self.cache['Z1'], derivative=True))
        dW1 = (1/ self.batchSize) * np.dot(dZ1, self.cache['X'])
        db1 = (1/ self.batchSize) * np.sum(dZ1, axis=1, keepdims=True)

        self.gradients = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

        return self.gradients


    def CELoss(self, y, output):
        lossSum = np.sum(np.multiply(y.T, np.log(output)))
        m = y.shape[0]
        loss = -(1. / m) * lossSum
        return loss

    def optimize(self):
        if self.optimizer == 'SGD':
            for key in self.params:
                self.params[key] = self.params[key] - self.eta * self.gradients[key]
        elif self.optimizer == "momentum":
            for key in self.params:
                self.momentums[key] = (self.momentum * self.momentums[key] + (1 - self.momentum) * self.gradients[key])
                self.params[key] = self.params[key] - self.eta * self.momentums[key]

    def accuracy(self, y, output):
        return np.mean(np.argmax(y, axis=-1) == np.argmax(output.T, axis=-1))

    def train(self, xTrain, yTrain, xTest, yTest):
        # Hyper-parameters
        labels = np.unique(yTrain)
        numClasses = len(labels)

        # ONE HOT ENCODING DOESN'T WORK
        # yTrain = self.oneHotEncoding(yTrain,numClasses)
        # yTest = self.oneHotEncoding(yTest,numClasses)
        # print(yTrain)

        # Initialize optimizer
        if self.optimizer == 'momentum':
            self.momentums = self.initMomentum()

        trainLosses = []
        testLosses = []

        trainAccuracies = []
        testAccuracies = []

        trainPredictions = []
        testPredictions = []

        for epoch in range(self.epochs):
            for index in range(0, xTrain.shape[0], self.batchSize):
                x = xTrain[index:index+self.batchSize]
                y = yTrain[index:index+self.batchSize]

                output = self.forward(x)
                grad = self.backProp(y, output)
                self.optimize()
            # Training data
            outputTrain = self.forward(xTrain)
            trainAccuracy = self.accuracy(yTrain, outputTrain)
            trainLoss = self.CELoss(yTrain, outputTrain)

            trainPredictions.append(outputTrain)
            trainLosses.append(trainLoss)
            trainAccuracies.append(trainAccuracy)

            # Test data
            outputTest = self.forward(xTest)
            testAccuracy = self.accuracy(yTest, outputTest)
            testLoss = self.CELoss(yTest, outputTest)

            testPredictions.append(outputTest)
            testLosses.append(testLoss)
            testAccuracies.append(testAccuracy)

            print("{:d}\t->\tTrainLoss : {:.7f}\t|\tTestLoss : {:.7f}\t|\tTrainAccuracy : {:.7f}\t|\tTestAccuracy: {:.7f}"
                  .format(epoch+1, trainLoss, testLoss, trainAccuracy, testAccuracy))

        # print('before one hot ',trainPredictions, np.shape(trainPredictions))
        # print(np.shape(trainPredictions))
        # print(trainPredictions[-1])
        lastTrainPredEnoc = self.oneHotEncoding(trainPredictions[-1], 2)
        # print('TrainP', lastTrainPredEnoc, np.shape(lastTrainPredEnoc))
        # testPredictions = self.oneHotEncoding(testPredictions,2)
        return trainLosses, testLosses, trainAccuracies, testAccuracies, trainPredictions[-1], testPredictions[-1]

def plotCM(trainPred, testPred, yTrain, yTest):
    # trainPred = [int(i) for i in trainPred]
    # yTrain = [int(i) for i in yTrain]
    # testPred = [int(i) for i in testPred]
    # yTest = [int(i) for i in yTest]
    x = [i for i in range(len(yTrain))]
    # plt.figure(figsize=(10, 7))
    # plt.scatter(x,yTrain, c='red')
    # plt.scatter(x, trainPred, c='green')
    # plt.show()
    trainPred = [(i>0.5).astype(int) for i in trainPred]
    # trainPred = [(trainPred>0.5).astype(int)]
    yTrain = np.array(yTrain, dtype=int)
    # testPred = np.array(testPred, dtype=int)
    testPred = [(i>0.5).astype(int) for i in testPred]
    yTest = np.array(yTest, dtype=int)
    trainPredS = pd.Series(trainPred, name='Predicted')
    yTrainS = pd.Series(yTrain.ravel().tolist(), name='Actual')


    # TRAINING CONFUSION MATRIX
    trainConfusionDF = pd.crosstab(yTrainS, trainPredS)
    trainCM = trainConfusionDF.to_numpy()
    # trainAccuracy = (trainCM[0][0] + trainCM[1][1]) / np.sum(trainCM.flatten())
    # trainPrecision = trainCM[1][1] / (trainCM[0][1] + trainCM[1][1])
    plt.figure(figsize= (10, 7))
    plt.suptitle('Training & Testing Confusion Matrix', fontsize = 20)
    plt.subplot(2,2,1)
    sns.heatmap(trainConfusionDF, annot=True)
    plt.subplot(2,2,2)
    sns.heatmap(trainCM / np.sum(trainCM), annot=True,
                fmt='.2%', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # traintxt = f'The Accuracy is {trainAccuracy}, and Precision is {trainPrecision}'
    # plt.figtext(0.5, 0.01, traintxt, wrap=True, horizontalalignment='center', fontsize=12)
    # plt.show()


    # TESTING CONFUSION MATRIX
    testPredS = pd.Series(testPred, name='Predicted')
    yTestS = pd.Series(yTest.ravel().tolist(), name='Actual')
    testConfusionDF = pd.crosstab(yTestS,testPredS)
    testCM = testConfusionDF.to_numpy()

    # testAccuracy = (testCM[0][0] + testCM[1][1])/ np.sum(testCM.flatten())
    # testPrecision = testCM[1][1] / (testCM[0][1] + testCM[1][1])
    # plt.figure(figsize=(10, 7))
    # plt.suptitle('Testing Confusion Matrix', fontsize = 20)
    plt.subplot(2,2,3)
    sns.heatmap(testConfusionDF, annot=True)
    plt.subplot(2,2,4)
    sns.heatmap(testCM / np.sum(testCM), annot=True,
                fmt='.2%', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # testtxt = f'The Accuracy is {testAccuracy}, and Precision is {testPrecision}'
    # plt.figtext(0.5, 0.01, testtxt, wrap=True, horizontalalignment='center', fontsize=12)
    # Precision is measured using TP/(TP+FP), Accuracy is measured using (TP+TN)/(TP+TN+FP+FN)
    plt.show()


def plotGraph(trainLosses, testLosses, trainAcc, testAcc):
    plt.subplot(1, 2, 1)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    plt.plot(trainLosses, label="Train loss")
    plt.plot(testLosses, label="Test loss")
    plt.legend(loc='best')
    plt.title("Epochs vs. CE Loss")
    plt.xlabel("Number of Epochs", fontsize = 10)
    plt.ylabel("Cross Entropy Loss", fontsize = 10)

    plt.subplot(1, 2, 2)
    plt.plot(trainAcc, label="Train Accuracy")
    plt.plot(testAcc, label="Test Accuracy")
    plt.legend(loc='best')
    plt.title("Epochs vs. Mean Class Accuracy")
    plt.xlabel("Number of Epochs", fontsize = 10)
    plt.ylabel("Mean Class Accuracy", fontsize = 10)
    plt.show()


def plotGraph2(trainLosses, testLosses, trainAcc, testAcc):
    length = len(trainLosses)

    fig, ax1= plt.subplots(1, length, sharex=True, figsize=(20, 12))
    fig.supxlabel('Number of Epochs', fontsize = 20)
    plt.suptitle('Cross Entropy Loss & Mean Classification Accuracy vs. Epochs for K(i) Fold', fontsize = 28)
    for index in range(length):
        ax1[index].plot(trainLosses[index], label="Train Loss")
        ax1[index].plot(testLosses[index], label="Test Loss")

        # ax2[index].plot(trainAcc[index], label="Train Accuracy", color='red')
        # ax2[index].plot(testAcc[index], label="Test Accuracy", color='purple')
        ax1[index].set_ylabel('CE Loss', fontsize= 17)
        ax1[index].legend(loc = 'upper right')
        # ax2[index].set_ylabel('Mean Classification Accuracy', fontsize = 17)
        # handles, labels = [(a + b) for a, b in zip(ax1[index].get_legend_handles_labels(), ax2[index].get_legend_handles_labels())]
        # fig.legend(handles, labels, loc='center right')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.85,
                        wspace=0.4,
                        hspace=0.2)
    plt.show()

    avgtrainAcc = np.mean(trainAcc, axis=0)
    avgtestAcc = np.mean(testAcc, axis=0)

    # plt.plot(avgtrainAcc, label="Avg. Train Accuracy")
    # plt.plot(avgtestAcc, label="Avg. Test Accuracy")
    # plt.legend(loc='best')
    # plt.title("Cross Validation Mean Class Accuracy vs. Epochs")
    # plt.xlabel("Number of Epochs", fontsize=10)
    # plt.ylabel("Mean Class Accuracy", fontsize=10)
    plt.show()


def DNNALGO(trainData, testData, NNSize, epochs, eta, batchSize, regStrength, momentum, activation, optimizer):
    xTrain = trainData[:, :-1]
    yTrain = trainData[:, -1]
    yTrain = yTrain.reshape((-1, 1))

    xTest = testData[:, :-1]
    yActual = testData[:, -1]
    yActual = yActual.reshape((-1, 1))

    nn = DNNClass(NNSize=NNSize, epochs=epochs, eta=eta, batchSize=batchSize, regStrength=regStrength, momentum=momentum, activation=activation, optimizer=optimizer)

    trainlosses, testlosses, trainaccuracy, testaccuracy, trainpred, testpred = nn.train(xTrain, yTrain, xTest, yActual)
    labels = np.unique(yTrain)
    numClasses = len(labels)

    YTrainEnoc = nn.oneHotEncoding(yTrain, numClasses)
    YTestEnoc = nn.oneHotEncoding(yActual, numClasses)
    trainPredenc = nn.oneHotEncoding(trainpred[-1], numClasses)
    testPredenc = nn.oneHotEncoding(testpred[-1], numClasses)
    # print('yTrain', yTrain[:5])
    # print('YTrainENC', YTrainEnoc[:5])
    # print('YTestENC', YTestEnoc[:5])
    #
    # print('trE', trainPredenc[:5])
    # print('TEE', testPredenc[:5])
    #trainlosses, testlosses, trainaccuracy, testaccuracy, trainpred[-1], testpred[-1], yTrain, yActual
    # DIDN"'t USE ONE HOT ENCODING
    # trainlosses, testlosses, trainaccuracy, testaccuracy, trainPredenc, testPredenc, YTrainEnoc, YTestEnoc
    return trainlosses, testlosses, trainaccuracy, testaccuracy, trainpred[-1], testpred[-1], yTrain, yActual


def CrossValidation(data, k, NNSize, epochs, eta, batchSize, regStrength, momentum, activation, optimizer):
    np.random.shuffle(data)
    folds = np.array_split(data, k)
    trainlosses = []
    testlosses = []
    trainaccuracy = []
    testaccuracy = []

    trainPreds = []
    testPreds = []
    yTrainList = []
    yActualList = []
    for i in range(k):
        testData = folds[i][:, :]
        newFolds = np.row_stack(np.delete(folds, i, 0))
        trainData = newFolds[:, :]
        print(f'Fold {i + 1}')
        # print(trainData)
        print(f'Testing set size : {testData.shape}')
        print(f'Training set size : {trainData.shape}\n')
        foldtrainloss, foldtestloss, foldtrainacc, foldtestacc, foldtrainPred, foldtestPred, foldyTrain, foldyActual = DNNALGO(trainData,
                                                    testData, NNSize, epochs, eta, batchSize, regStrength, momentum, activation, optimizer)
        trainlosses.append(foldtrainloss)
        testlosses.append(foldtestloss)

        trainaccuracy.append(foldtrainacc)
        testaccuracy.append(foldtestacc)
        trainPreds.extend(foldtrainPred)
        testPreds.extend(foldtestPred)

        yTrainList.extend(foldyTrain)
        yActualList.extend(foldyActual)
    # print('Yactual')
    # print(yActualList)
    # print(len(yActualList))
    # print('YPred')
    # print(yTrainList)
    # print(len(yTrainList))
    # yTrainList = np.concatenate(yTrainList).ravel().tolist()
    # yActualList = np.concatenate(yActualList).ravel().tolist()
    plotGraph2(trainlosses, testlosses, trainaccuracy, testaccuracy)
    plotCM(trainPreds,testPreds,yTrainList,yActualList)
    return


# Testing on Dummy Data
xTrain = np.asarray(xTrain)
xTest = np.asarray(xTest)
yTrain = np.asarray(yTrain)
yTest = np.asarray(yTest)

# dnn.train(xTrain, yTrain, xTest, yTest, batchSize=2, optimizer='momentum', eta=0.4, beta=.9)
# dnn = DNNClass(NNSize= [3, 3, 3, 1], epochs=100, eta=0.02, batchSize=2, regStrength=0.001, momentum=0.05, activation='sigmoid', optimizer='momentum')
# t1,t2,t3,t4,t5,t6 = dnn.train(xTrain, yTrain, xTest, yTest)
# dummyTotalData = np.array(dummyTotalData)
# CrossValidation(dummyTotalData, 2, [3,3,3,1],3, 0.02, 2, 0.01, 0.05, 'sigmoid', 'momentum')