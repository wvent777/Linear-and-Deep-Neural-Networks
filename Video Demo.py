import DataPreprocessing as dp
import LinearRegression as LR
import DNNReg as DR
import LinearClassification as LC
import DNNClass as DC
import AutoEncoder as AE
import pandas as pd
import numpy as np

def normalize(data):
    for column in data.columns:
        data[column] = 2 * ((data[column] - data[column].min())/(data[column].max() - data[column].min())) - 1
    return data

# Showing both here
# – Demonstrate the weight updates for logistic regression.
# – Demonstrate the gradient calculation at the output for one of your networks.
# Note - Activation for Linear network shown here too
dummydata = [[0.234, 0.422, 0.543, 1],
             [0.562, 0.75, 0.35, 0],
             [0.7513, 0.21, 0.012, 1],
             [0.943, 0.4, 0.75, 0],
             [0.8, 0.2, 0.01, 0],
             [0.93, 0.7, 0.3, 1]]

# test1 = LC.CrossValidation(dummydata, 2, 2, 0.02, 3, 0.001, 0.05)



# - Demonstrate and explain how an example is propagated through each network.
# Be sure to show the activations at each layer being calculated correctly.
# Deep Neural Network

test4 = DC.DNNClass([3,3,3,1], 2, 0.02, 1, 0.001, 0.05, 'sigmoid', 'SDG')
xTrain = [[0.234,0.422,0.543]]
xTrain= np.asarray(xTrain)
# test4.forward(xTrain)



# – Demonstrate the weight updates occurring on a two-layer network for each of the layers.
dummy2train = [[9, 30, 3, 184],
              [5, 44, 4, 146],
              [16, 41, 1, 91],
              [7, 22, 4, 209],
              [13, 34, 6, 132],
              [7, 30, 0, 55],
              [1, 19, 6, 56]]

dummy2train = pd.DataFrame(dummy2train, columns=['X1', 'X2', 'X3', 'Y'])
normalize(dummy2train)
dummy2train = dummy2train.to_numpy()
test3 = DR.DNNReg([3,4,3,1], epochs=2, eta=0.002, regStrength=0.001, momentum=0.05)
dummy2np = np.array(dummy2train)
dummy2XTrain = dummy2np[:4, :-1]
dummy2XTest = dummy2np[4:-1, :-1]
dummy2YTrain = dummy2np[:4, -1]
dummy2YTest = dummy2np[4:-1, -1]
# trainLosses, testLosses, trainMSE, testMSE, trainPred, testPred = test3.train(dummy2XTrain, dummy2YTrain, dummy2XTest, dummy2YTest)




# Showing Both here
# – Demonstrate the weight updates occurring during the training of the autoencoder.
# – Demonstrate the autoencoding functionality (i.e., recovery of an input pattern on the output).
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
test2 = AE
test2.autoencoder([3,2,3], 100, 0.02).train(xNP)
