import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dummydata = [[0.234,0.422,0.543,1],
[0.562,0.75,0.35,0],
[0.7513,0.21,0.012,1],
[0.943,0.4,0.75,0],]

class DNNClass:
    def __init__(self, numPerLayer, epochs, eta, regStrength, momentum):
        """
         :param epochs: No. of iterations over entire training data
         :param eta: learning rate value
         :param regStrength: L2 Regularization Strength
         :param momentum: Momentum Value
         """
        self.numPerLayer = numPerLayer
        self.epochs = epochs
        self.eta = eta
        self.regStrength = regStrength
        self.momentum = momentum
        self.velocity = None
        self.weights = []
        self.bias = []

        layers = len(numPerLayer)
        for layer in range(layers - 1):
            # He-et-al Weight initialization
            self.weights.append(np.random.rand(numPerLayer[layer + 1], numPerLayer[layer]) * np.sqrt(2 / numPerLayer[layer]))
        for layer in range(1, layers):  # Generating bias except for input layer
            self.bias.append(np.random.rand(numPerLayer[layer]))
        self.activityList = []
        self.activationList = []

    def oneHotEncoding(self, y, numOfClasses):
        y = np.asarray(y, dtype='int32')
        if len(y) > 1:
            y = y.reshape(-1)
        if not numOfClasses:
            numOfClasses = np.max(y) + 1
        yMatrix = np.zeros((len(y), numOfClasses))
        yMatrix[np.arange(len(y)), y] = 1
        return yMatrix


    def softmax(self, values):
        values -= np.max(values)
        probability = (np.exp(values).T / np.sum(np.exp(values), axis=0)).T
        return probability

    def CELoss2(self, X, yMatrix):
        output = None
        print('This is the output')
        print(output)
        for weight, bias in zip(self.weights,self.bias):
            activity = np.dot(X, weight.T) + bias
            self.activityList.append(activity)
            X = self.softmax(activity)
            self.activationList.append(X)
            output = X
            print(output)
        deltaCE = -np.log(np.max(output))*yMatrix
        self.error = (1/2) * np.power(yMatrix-output, 2)
        self.deltaE = (yMatrix-output)

        return output, deltaCE

    def predict2(self, X):
        predictions = []
        for x in X:
            output = 0
            for weight, bias in zip(self.weights,self.bias):
                activity = np.dot(x, weight.T) + bias
                x = self.softmax(activity)
                output = x
            predictions.append(output)
        predictions = np.ravel(predictions) # double Check
        return predictions

    def predict(self, x):
        return np.argmax(x.dot(self.weights), 1)

    def StochasticGD(self, X, Y):
        losses = []
        print('These are the weights')
        print(self.weights)
        output, dCE = self.CELoss2(X, Y)
        deltaCE = -np.log(np.max(output))*Y
        delta = np.dot(self.deltaE , dCE)


        self.weights[-1] -= self.eta * (self.activationList[-1] * delta)
        self.bias[-1] -= self.eta * delta

        for (index, array), activation in zip(reversed(list(enumerate(self.activationList[:-1]))), reversed(self.activityList[:-1])):
            print(index)
            print('activation', activation)
            delta = np.dot(self.weights[index+1].T, delta) * (-np.log(np.max(activation)))

            self.weights[index] -= self.eta * np.dot(delta.T, array)
            self.bias[index] -= self.eta * delta.ravel()
        return

    def meanAccuracy(self, x, y):
        ypred = self.predict(x)
        ypred = ypred.reshape((-1, 1))  # convert to column vector
        return np.mean(np.equal(y, ypred))

    def train(self, xTrain, yTrain, xTest, yTest):
        labels = np.unique(yTrain)
        numClasses = len(labels)

        yTrainEncoded = self.oneHotEncoding(yTrain, numClasses)
        yTestEncoded = self.oneHotEncoding(yTest, numClasses)
        self.velocity = []
        for weight in self.weights:
            layerVelocity = np.zeros(weight.shape)
            self.velocity.append(layerVelocity)
        self.biasVelocity = []
        for bias in self.bias:
            biasVLayer = np.zeros(bias.shape)
            self.biasVelocity.append(biasVLayer)
        trainLosses = []
        testLosses = []
        trainAccuracy = []
        testAccuracy = []
        trainPredicted = []
        testPredicted = []
        for epoch in range(self.epochs):
            for xtrain, ytrainenc in zip(xTrain,yTrainEncoded):
                print('This is our X examined')
                print(xtrain)
                print('this is our Yencoded')
                print(ytrainenc)
                self.StochasticGD(xtrain, ytrainenc)


            # testLoss, deltaW, deltaWb = self.CELoss(xTest, yTestEncoded)
            #
            # trainPredicted.append(self.predict(xTrain))
            # testPredicted.append(self.predict(xTest))
            #
            # trainAccuracy.append(self.meanAccuracy(xTrain, yTrain))
            # testAccuracy.append(self.meanAccuracy(xTest, yTest))
            #
            # trainLosses.append(trainLoss)
            # testLosses.append(testLoss)
            # print("{:d}\t->\tTrainL : {:.7f}\t|\tTestL : {:.7f}\t|\tTrainAcc : {:.7f}\t|\tTestAcc: {:.7f}"
            #       .format(epoch, trainLoss, testLoss, trainAccuracy[-1], testAccuracy[-1]))

        return trainLosses, testLosses, trainAccuracy, testAccuracy, trainPredicted[-1], testPredicted[-1]



def plotCM(trainPred, testPred, yTrain, yTest):
    trainPredS = pd.Series(trainPred, name='Predicted')
    yTrainS = pd.Series(yTrain, name='Actual')
    trainConfusionDF = pd.crosstab(yTrainS, trainPredS)
    trainCM = trainConfusionDF.to_numpy()
    trainAccuracy = (trainCM[0][0] + trainCM[1][1]) / np.sum(trainCM.flatten())
    trainPrecision = trainCM[1][1] / (trainCM[0][1] + trainCM[1][1])

    plt.figure(figsize= (10, 7))
    plt.suptitle('Training Confusion Matrix', fontsize = 20)
    plt.subplot(1,2,1)
    sns.heatmap(trainConfusionDF, annot=True)
    plt.subplot(1,2,2)
    sns.heatmap(trainCM / np.sum(trainCM), annot=True,
                fmt='.2%', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    traintxt = f'The Accuracy is {trainAccuracy}, and Precision is {trainPrecision}'
    plt.figtext(0.5, 0.01, traintxt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()

    testPredS = pd.Series(testPred, name='Predicted')
    yTestS = pd.Series(yTest, name='Actual')
    testConfusionDF = pd.crosstab(yTestS,testPredS)
    testCM = testConfusionDF.to_numpy()

    testAccuracy = (testCM[0][0] + testCM[1][1])/ np.sum(testCM.flatten())
    testPrecision = testCM[1][1] / (testCM[0][1] + testCM[1][1])
    plt.figure(figsize=(10, 7))
    plt.suptitle('Testing Confusion Matrix', fontsize = 20)
    plt.subplot(1,2,1)
    sns.heatmap(testConfusionDF, annot=True)
    plt.subplot(1,2,2)
    sns.heatmap(testCM / np.sum(testCM), annot=True,
                fmt='.2%', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    testtxt = f'The Accuracy is {testAccuracy}, and Precision is {testPrecision}'
    plt.figtext(0.5, 0.01, testtxt, wrap=True, horizontalalignment='center', fontsize=12)
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

    fig, (ax1, ax2) = plt.subplots(2, length, sharex=True, figsize=(20, 12), gridspec_kw={'height_ratios': [1, 1]})
    fig.supxlabel('Number of Epochs', fontsize = 20)
    plt.suptitle('Cross Entropy Loss & Mean Classification Accuracy vs. Epochs for K(i) Fold', fontsize = 28)
    for index in range(length):
        ax1[index].plot(trainLosses[index], label="Train Loss")
        ax1[index].plot(testLosses[index], label="Test Loss")

        ax2[index].plot(trainAcc[index], label="Train Accuracy", color='red')
        ax2[index].plot(testAcc[index], label="Test Accuracy", color='purple')
        ax1[index].set_ylabel('CE Loss', fontsize= 17)
        ax2[index].set_ylabel('Mean Classification Accuracy', fontsize = 17)
        handles, labels = [(a + b) for a, b in zip(ax1[index].get_legend_handles_labels(), ax2[index].get_legend_handles_labels())]
        fig.legend(handles, labels, loc='center right')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.85,
                        wspace=0.4,
                        hspace=0.2)
    plt.show()

    avgtrainAcc = np.mean(trainAcc, axis=0)
    avgtestAcc = np.mean(testAcc, axis=0)

    plt.plot(avgtrainAcc, label="Avg. Train Accuracy")
    plt.plot(avgtestAcc, label="Avg. Test Accuracy")
    plt.legend(loc='best')
    plt.title("Cross Validation Mean Class Accuracy vs. Epochs")
    plt.xlabel("Number of Epochs", fontsize=10)
    plt.ylabel("Mean Class Accuracy", fontsize=10)
    plt.show()


def LinearSoftmaxAlgo(trainData, testData, nnlayers, epochs, eta, regStrength, momentum):
    xTrain = trainData[:, :-1]
    yTrain = trainData[:, -1]
    yTrain = yTrain.reshape((-1, 1))

    xTest = testData[:, :-1]
    yActual = testData[:, -1]
    yActual = yActual.reshape((-1, 1))

    nn = DNNClass(numPerLayer=nnlayers, epochs=epochs, eta=eta, regStrength=regStrength, momentum=momentum)
    trainlosses, testlosses, trainaccuracy, testaccuracy, trainpred, testpred = nn.train(xTrain, yTrain, xTest, yActual)
    return trainlosses, testlosses, trainaccuracy, testaccuracy, trainpred, testpred, yTrain, yActual


def CrossValidation(data, k, nnlayers, epochs, eta, regStrength, momentum):
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
        print(f'Testing set size : {testData.shape}')
        print(f'Training set size : {trainData.shape}\n')
        foldtrainloss, foldtestloss, foldtrainacc, foldtestacc, foldtrainPred, foldtestPred, foldyTrain, foldyActual = LinearSoftmaxAlgo(trainData,
                                                    testData, nnlayers, epochs, eta, regStrength, momentum)
        trainlosses.append(foldtrainloss)
        testlosses.append(foldtestloss)

        trainaccuracy.append(foldtrainacc)
        testaccuracy.append(foldtestacc)

        trainPreds.extend(foldtrainPred)
        testPreds.extend(foldtestPred)

        yTrainList.extend(foldyTrain)
        yActualList.extend(foldyActual)
    yTrainList = np.concatenate(yTrainList).ravel().tolist()
    yActualList = np.concatenate(yActualList).ravel().tolist()
    plotGraph2(trainlosses, testlosses, trainaccuracy, testaccuracy)
    plotCM(trainPreds,testPreds,yTrainList,yActualList)
    return

CrossValidation(dummydata, 5, [3,4,3,1], 5, 0.002 , 0.001, 0.05)
