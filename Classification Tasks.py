import LinearClassification
import DataPreprocessing as dp
import DNNClass as DC
import AutoEncoder as AE

# Breast Classification
etavalues = [0.01, 0.015, 0.02, 0.03, 0.035]
# for x in etavalues:
#     breastTuning = LinearClassification.CrossValidation(dp.breastTuningNP, 5, 300, x, 3, 0.001, 0.05)

# eta = 0.01 gives the highest accuracy and precision on the testing

# breastTesting = LinearClassification.CrossValidation(dp.breastNP80, 5, 300, 0.01, 3, 0.001, 0.05)

# breastTesting = FFBPC.CrossValidation(dp.breastTuningNP,5, [9,10,9,1],5, 0.01, 0.001, 0.05)

# Deep Neural Network
# Neural Network Architecture 9 Input -> 9 Hidden 1 Inputs -> 9 Hidden 2 Inputs - > 1 Output
# for eta in etavalues:
#     breastTuning = DC.CrossValidation(dp.breastTuningNP,5,[9,9,9,1],500, eta, 10, 0.001, 0.5,'sigmoid', 'momentum')
# DNN Testing
# breastDNNtest =  DC.CrossValidation(dp.breastNP80, 5, [9,9,9,1], 500, 0.015, 10, 0.001, 0.05, 'sigmoid', 'momentum')

# AUTOENCODER THEN TO NETWORK
# Autoencoder Network Architecture 9 Input -> 8 Hidden 1 Inputs -> 9 Output
# DNN Network Architecture 8 Inputs -> 8 H1 Nodes -> 1 Output
# for eta in etavalues:
#     breastAENN = AE.AENN(dp.breastTuningNP, 5, 500, eta, 0.001, 0.05)
#     breastAENN.fitCLASS(dp.breastTuningNP)

# Testing on 80%
# Best accuracy at eta = 0.01 with accuracy being 0.8214
# breastAENNTest= AE.AENN(dp.breastNP80, 5, 500, 0.01, 0.001, 0.05)
# breastAENNTest.fitCLASS(dp.breastNP80)



# HOUSE CLASSIFICATION
# for eta in etavalues:
#     votingTuning = LinearClassification.CrossValidation(dp.houseTuningNP, 5, 300, eta, 3, 0.001, 0.05)

# eta = 0.03 or 0.035 gives the highest accuracy and precision on the testing

# votingTesting = LinearClassification.CrossValidation(dp.houseNP80, 5, 300, 0.03, 3, 0.001, 0.05)

# Deep Neural Network
# Neural Network Architecture 16 Input -> 16 Hidden 1 Inputs -> 16 Hidden 2 Inputs - > 1 Output
# for eta in etavalues:
#     votingTuning = DC.CrossValidation(dp.houseTuningNP,5,[16,16,16,1],500, eta, 10, 0.001, 0.5,'sigmoid', 'momentum')

# voteDNNTEST = DC.CrossValidation(dp.houseNP80,5,[16,16,16,1],500, 0.02, 10, 0.001, 0.5,'sigmoid', 'momentum')

# AUTOENCODER THEN TO NETWORK
# Autoencoder Network Architecture 9 Input -> 8 Hidden 1 Inputs -> 9 Output
# DNN Network Architecture 8 Inputs -> 8 H1 Nodes -> 1 Output
# for eta in etavalues:
#     houseAENN = AE.AENN(dp.houseTuningNP, 5, 500, eta, 0.001, 0.05)
#     houseAENN.fitCLASS(dp.houseTuningNP)
#
# houseAENNTest= AE.AENN(dp.houseNP80, 5, 500, 0.03, 0.001, 0.05)
# houseAENNTest.fitCLASS(dp.houseNP80)


# Car Classification
# for eta in etavalues:
#     carTuning = LinearClassification.CrossValidation(dp.carTuningNP, 5, 300, eta, 3, 0.001, 0.05)

# carTesting = LinearClassification.CrossValidation(dp.carNP80, 5, 300, 0.01, 3, 0.001, 0.05)

# Deep Neural Network
# Neural Network Architecture 6 Input -> 6 Hidden 1 Inputs -> 6 Hidden 2 Inputs - > Output
# for eta in etavalues:
#     carTuning = DC.CrossValidation(dp.carTuningNP,5,[6,6,6,1],500, eta, 10, 0.001, 0.5,'sigmoid', 'momentum')

# carTesting = DC.CrossValidation(dp.carNP80,5,[6,6,6,1],500, 0.03, 10, 0.001, 0.5,'sigmoid', 'momentum')


# AUTOENCODER THEN TO NETWORK
# Autoencoder Network Architecture 6 Input -> 5 Hidden 1 Inputs -> 6 Output
# DNN Network Architecture 5 Inputs -> 4 H1 Nodes -> 1 Output
# for eta in etavalues:
#     carAENN = AE.AENN(dp.carTuningNP, 5, 500, eta, 0.001, 0.05)
#     carAENN.fitCLASS(dp.carTuningNP)

# Best Accuracy at 0.6560 with eta = 0.015

# carAENNTest= AE.AENN(dp.carNP80, 5, 500, 0.015, 0.001, 0.05)
# carAENNTest.fitCLASS(dp.carNP80)
