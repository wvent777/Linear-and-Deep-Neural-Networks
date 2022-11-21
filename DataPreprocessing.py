import numpy as np
import pandas as pd

def normalize(data):
    for column in data.columns:
        data[column] = 2 * ((data[column] - data[column].min())/(data[column].max() - data[column].min())) - 1
    return data
def impute(data, type):
    for column in data.columns:
        mean = data[column].mean()
        if type =='float':
            data[column] = data[column].fillna(mean)
        else:
            data[column] = data[column].fillna(mean).astype(int)
    return data
def ordinalEncode(dataset, value_map):
    for column in dataset.columns:
        lst = []
        for index in dataset[column]:
            if index in value_map.keys():
                value = value_map.get(index)
                lst.append(value)
            else:
                lst.append(index)
                continue
        dataset[column] = lst
    return dataset


# Regression Problems   - Normalize all of Regression Problems
# Abalone - (Rings is Value to Predict)
abaloneData = pd.read_csv('data/abalone.data', names= ['Sex', 'Length', 'Diameter',
                                                        'Height', 'WholeWeight', 'ShuckedWeight',
                                                        'VisceraWeight', 'ShellWeight', 'Rings'])
abaloneDF = pd.DataFrame(abaloneData)
abaloneDF['Sex'] = abaloneDF['Sex'].astype('category')
abaloneDF['Sex'] = abaloneDF['Sex'].cat.codes
abaloneDF = normalize(abaloneDF)

abaloneDF80 = abaloneDF.sample(frac=0.8)
abaloneTuningDF = abaloneDF.drop(abaloneDF80.index)
abaloneDF80 = abaloneDF80.reset_index(drop=True)
abaloneTuningDF = abaloneTuningDF.reset_index(drop=True)
abaloneNP80 = abaloneDF80.to_numpy()
abaloneTuningNP = abaloneTuningDF.to_numpy()


# Computer Hardware (ERP - Last One is the Estimated)
computerData = pd.read_csv('data/machine.data', names=['vendor', 'model', 'MYCT',
                                                       'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX',
                                                       'PRP', 'ERP'])
computerDF = pd.DataFrame(computerData.iloc[:, :-1])
computerDF['vendor'] = computerDF['vendor'].astype('category')
computerDF['vendor'] = computerDF['vendor'].cat.codes
computerDF['model'] = computerDF['model'].astype('category')
computerDF['model'] = computerDF['model'].cat.codes
computerDF = normalize(computerDF)

computerDF80 = computerDF.sample(frac=0.8)
computerTuningDF = computerDF.drop(computerDF80.index)
computerDF80 = computerDF80.reset_index(drop=True)
computerTuningDF = computerTuningDF.reset_index(drop=True)
computerNP80 = computerDF80.to_numpy()
computerTuningNP = computerTuningDF.to_numpy()

# Forest Fire
forestData = pd.read_csv('data/forestfires.data', header=0)
forestData.month = forestData.month.astype('category')
forestData.month = forestData.month.cat.codes
forestData.day = forestData.day.astype('category')
forestData.day = forestData.day.cat.codes
# Added plus 1 to handle the -inf problem
forestData.area = np.log(forestData.area + 1)
forestDF = pd.DataFrame(forestData)
forestDF = normalize(forestDF)

forestDF80 = forestDF.sample(frac=0.8)
forestTuningDF = forestDF.drop(forestDF80.index)
forestDF80 = forestDF80.reset_index(drop=True)
forestTuningDF = forestTuningDF.reset_index(drop=True)
forestNP80 = forestDF80.to_numpy()
forestTuningNP = forestTuningDF.to_numpy()


# Classification
# Breast Cancer - Removing the sample column
breastData = pd.read_csv('data/breast-cancer-wisconsin.data', names=['sample', 'ClumpThickness', 'UCellSize',
                                                                     'UCellShape', 'MA', 'SECellSize',
                                                                     'BareNuclei', 'BlandChromatin', 'NormalNucleoli',
                                                                     'Mitosis', 'Class'], na_values=['?'])
breastDF = pd.DataFrame(breastData.iloc[:, 1:])
breastDF = impute(breastDF, 'int')
breastDF['Class'] = breastDF['Class'].astype('category')
breastDF['Class'] = breastDF['Class'].cat.codes

breastDF80 = breastDF.sample(frac=0.8)
breastTuningDF = breastDF.drop(breastDF80.index)

breastDF80 = breastDF80.reset_index(drop=True)
breastTuningDF = breastTuningDF.reset_index(drop=True)

breastNP80 = breastDF80.to_numpy()
breastTuningNP = breastTuningDF.to_numpy()


# Car Data
carData = pd.read_csv("data/car.data", names= ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acc'])
carDF = pd.DataFrame(carData)
carDF.buying = pd.Categorical(carDF.buying, ['low', 'med', 'high' 'vhigh'], ordered=True)
carDF.buying = carDF.buying.cat.codes
carDF.maint = pd.Categorical(carDF.maint, ['low', 'med', 'high' 'vhigh'], ordered=True)
carDF.maint = carDF.maint.cat.codes
carDF.doors = pd.Categorical(carDF.doors, ['2', '3', '4', '5more'], ordered=True)
carDF.doors = carDF.doors.cat.codes
carDF.persons = pd.Categorical(carDF.persons, ['2', '4', 'more'], ordered=True)
carDF.persons = carDF.persons.cat.codes
carDF.lug_boot = pd.Categorical(carDF.lug_boot, ['small', 'med', 'big'], ordered=True)
carDF.lug_boot = carDF.lug_boot.cat.codes
carDF.safety = pd.Categorical(carDF.safety, ['low', 'med', 'high'], ordered=True)
carDF.safety = carDF.safety.cat.codes
carDF.acc = pd.Categorical(carDF.acc, ['unacc', 'acc', 'good','vgood'], ordered=True)
carDF.acc = carDF.acc.cat.codes

carDF_80 = carDF.sample(frac=0.8)
carTuningDF = carDF.drop(carDF_80.index)

carTuningDF = carTuningDF.reset_index(drop=True)
carDF_80 = carDF_80.reset_index(drop=True)
carTuningNP = carTuningDF.to_numpy()
carNP80 = carDF_80.to_numpy()



# House Vote
houseData = pd.read_csv('data/house-votes-84.data', names=['Class', 'HI', 'WPCS', 'ABR',
                                                           'PFF', 'ESA', 'RGS', 'ASTB', 'ANC',
                                                           'MM', 'IM', 'SCC', 'ES', 'SRS',
                                                           'CR', 'DFE', 'SA'], na_values=['?'])
houseData = houseData.reindex(columns=[ 'HI', 'WPCS', 'ABR',
                                    'PFF', 'ESA', 'RGS', 'ASTB', 'ANC',
                                    'MM', 'IM', 'SCC', 'ES', 'SRS',
                                    'CR', 'DFE', 'SA', 'Class'])
votemap = {"democrat": 0, "republican": 1,"y": 1, "n": 0}

ordinalEncode(houseData, votemap)
impute(houseData, 'int')
houseDF = pd.DataFrame(houseData)
houseDF80 = houseDF.sample(frac=0.8)
houseTuningDF = houseDF.drop(houseDF80.index)

houseDF80 = houseDF80.reset_index(drop=True)
houseTuningDF = houseTuningDF.reset_index(drop=True)

houseNP80 = houseDF80.to_numpy()
houseTuningNP = houseTuningDF.to_numpy()