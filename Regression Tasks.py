import DataPreprocessing as dp
import LinearRegression as LR
import DNNReg as NNReg
import AutoEncoder as AE


# ABALONE
etavalues = [0.01, 0.015, 0.02, 0.03, 0.035]
# Linear Neural Network
# Tuning:
# for eta in etavalues:
#     abaloneTuning = LR.CrossValidation(dp.abaloneTuningNP, 5, 500, eta, 5, 0.001, 0.05)
# Testing:Eta = 0.171
# abaloneTesting = LR.CrossValidation(dp.abaloneNP80, 5, 500, 0.035, 5, 0.001, 0.05)

# Deep Neural Network w/ hidden layers
# 8 Inputs -> 1 Ouput
# Neural Network Architecture -> [8,9,8,1]
# Tuning:
# for eta in etavalues:
#     abaloneTuning = NNReg.CrossValidation(dp.abaloneTuningNP,5, [8,9,8,1], 500, eta, 0.001, 0.05)
# abaloneTesting = NNReg.CrossValidation(dp.abaloneNP80, 5, [8, 9, 8,1], 500, 0.035, 0.001, 0.05)

# AUTOENCODER THEN TO NETWORK
# Autoencoder Network Architecture 9 Input -> 8 Hidden 1 Inputs -> 9 Output
# DNN Network Architecture 8 Inputs -> 9 H1 Nodes -> 8 H2 Nodes -> 1 Ouput
# for eta in etavalues:
#     abaloneAENN = AE.AENN(dp.abaloneTuningNP, 5, 200, eta, 0.001, 0.05)
#     abaloneTuningAE = abaloneAENN.fitREG(dp.abaloneTuningNP)

# Best Eta is 0.015
# abaloneTestAENN = AE.AENN(dp.abaloneNP80, 5, 200, 0.015, 0.001, 0.05)
# abaloneTestAENN.fitREG(dp.abaloneNP80)


# FOREST FIRE
forestetavalues = [0.01, 0.0125, 0.015, 0.0175, 0.02]
# Linear Neural Network
# Tuning:

# for eta in forestetavalues:
#     forestTuning = LR.CrossValidation(dp.forestTuningNP,5,500, eta, 5, 0.001, 0.05)
# Testing:
# forestTesting = LR.CrossValidation(dp.forestNP80, 5, 500, 0.2, 5, 0.001, 0.05)

# Deep Neural Network w/ hidden layers
# 12 Inputs -> 1 Output
# Neural Network Architecture -> [12,13,12,1]
# Tuning:
# for eta in forestetavalues:
#     forestNNTuning = NNReg.CrossValidation(dp.forestTuningNP, 5, [12,13,12,1], 500, eta, 0.001, 0.05)

# forestNNTest = NNReg.CrossValidation(dp.forestNP80, 5, [12,13,12,1], 500, 0.017, 0.001, 0.05)

# AUTOENCODER THEN TO NETWORK
# Autoencoder Network Architecture 12 Input -> 11 Hidden 1 Inputs -> 12 Output
# DNN Network Architecture 11 Inputs -> 12 H1 Nodes -> 11 H2 Nodes -> 1 Output
# for eta in forestetavalues:
#     forestAENN = AE.AENN(dp.forestTuningNP, 5, 200, eta, 0.001, 0.05)
#     forestAENN.fitREG(dp.forestTuningNP)

# Best Eta is 0.02
# forestTestAENN = AE.AENN(dp.forestNP80, 5, 200, 0.02, 0.001, 0.05)
# forestTestAENN.fitREG(dp.forestNP80)



# COMPUTER HARDWARE
computeretavalues = [0.01, 0.0125, 0.015, 0.0175, 0.02]
# Linear Neural Network
# Tuning:
# for eta in computeretavalues:
#     computerTuning = LR.CrossValidation(dp.computerTuningNP,5, 100, eta, 3, 0.001, 0.05)
# Testing:
# computerTesting = LR.CrossValidation(dp.computerNP80,5, 100, 0.0125, 3, 0.001, 0.05)

# Deep Neural Network w/ hidden layers
# 8 Inputs -> 1 Ouput
# Neural Network Architecture -> [8,9,8,1]
# Tuning:
# for eta in computeretavalues:
#     computerNNTuning = FFBBReg.CrossValidation(dp.computerTuningNP,5, [8,9,8,1], 100, eta, 0.001, 0.05)

# Testing:
# computerTesting = FFBBReg.CrossValidation(dp.computerNP80, 5, [8,9,8,1], 100, 0.015, 0.001, 0.05)

# AUTOENCODER THEN TO NETWORK
# Autoencoder Network Architecture 8 Input -> 7 Hidden 1 Inputs -> 8 Output
# DNN Network Architecture 7 Inputs -> 8 H1 Nodes -> 8 H2 Nodes -> 1 Output
# for eta in computeretavalues:
#     computerAENN = AE.AENN(dp.computerTuningNP, 5, 200, eta, 0.001, 0.05)
#     computerAENN.fitREG(dp.computerTuningNP)

# # Best Eta is 0.0125
computerTestAENN = AE.AENN(dp.computerNP80, 5, 200, 0.0125, 0.001, 0.05)
computerTestAENN.fitREG(dp.computerNP80)

