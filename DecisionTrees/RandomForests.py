import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

testDataFiles  = ["test_c300_d100","test_c300_d1000","test_c300_d5000","test_c500_d100","test_c500_d1000","test_c500_d5000","test_c1000_d100","test_c1000_d1000","test_c1000_d5000","test_c1500_d100","test_c1500_d1000","test_c1500_d5000","test_c1800_d100","test_c1800_d1000","test_c1800_d5000"]
validDataFiles = ["valid_c300_d100","valid_c300_d1000","valid_c300_d5000","valid_c500_d100","valid_c500_d1000","valid_c500_d5000","valid_c1000_d100","valid_c1000_d1000","valid_c1000_d5000","valid_c1500_d100","valid_c1500_d1000","valid_c1500_d5000","valid_c1800_d100","valid_c1800_d1000","valid_c1800_d5000"]
trainDataFiles = ["train_c300_d100","train_c300_d1000","train_c300_d5000","train_c500_d100","train_c500_d1000","train_c500_d5000","train_c1000_d100","train_c1000_d1000","train_c1000_d5000","train_c1500_d100","train_c1500_d1000","train_c1500_d5000","train_c1800_d100","train_c1800_d1000","train_c1800_d5000"]


# Returns split data without target column(pure 'X' data) & Target column data(pure 'y' Data)
def data(fileName):
    relativePath ="./Data/"
    filePath=relativePath+fileName+".csv"
    data = pd.read_csv(filePath, header=None)

    dataRows = data.shape[0]
    dataCols = data.shape[1]

    # print(dataRows,dataCols)

    y=data[data.columns[-1]]
    x=data[data.columns[0:dataCols-1]]

    return x,y

# Count number of files
files = testDataFiles.__len__()

resultsMatrix={}
trainedModelsList=[]


for file in range(0,files):

    x_train,y_train = data(trainDataFiles[file])
    x_test,y_test = data(testDataFiles[file])
    # print("filename : ",testDataFiles[file])
    # print(x_train.head(),y_train.head())


    clf = RandomForestClassifier()

    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)

    # print("File : ",testDataFiles[file],"  Accuracy : ",metrics.accuracy_score(y_test,y_pred))
    trainedModelsList.append(clf)
    resultsMatrix[testDataFiles[file]]= metrics.accuracy_score(y_test,y_pred)


f = open( 'ResultsMatrix-RandomForests.txt', 'w' )
f.write( 'dict = ' + repr(resultsMatrix) + '\n' )
f.close()

print("----- Accuracies ----- ")
for result in resultsMatrix:
    print(result," : ",resultsMatrix[result])
