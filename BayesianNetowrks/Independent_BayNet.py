import csv
import math

import numpy as np
import timeit

start = timeit.default_timer()

root_path = "C:\\Users\\darwi\\OneDrive - The University of Texas at Dallas\\Acads\\Machine Learning\\Assignments\\MachineLearning\\Data\\small-10-datasets\\"
train_file = "accidents.ts"
test_file = "accidents.test"
extension = ".data"

x_train = np.loadtxt(root_path + train_file + ".data", delimiter=',', dtype=int)
x_test = np.loadtxt(root_path + test_file + ".data", delimiter=',', dtype=int)

n_size = x_train.shape[0]
n_features = x_train.shape[1]
print("Size : ", n_size, " X ", n_features)

#
# def extractData(fileName):
#     # with open(fileName,newline='', encoding='utf_8') as csvfile:
#     with open(fileName, 'rt') as csvfile:
#         reader = csv.reader(csvfile, delimiter=',')
#         rows = list(reader)
#     NumberOfExamples = len(rows)
#     NumberOfFeatures = len(rows[0])
#     X = np.zeros((NumberOfExamples, NumberOfFeatures), dtype=int)
#     for i in range(0, NumberOfExamples):
#         X[i, :] = rows[i]
#     return X
#
#
# def findingParameters(X):
#     NumberOfTrainingExamples = len(X)
#     NumberOfFeatures = len(X[0, :])
#     ParameterArray = np.zeros((2, NumberOfFeatures))
#     den = float(NumberOfTrainingExamples + 2)
#     for j in range(0, NumberOfFeatures):
#         num_ones = (X[:, j] == 1).sum() + 1
#         ParameterArray[1, j] = num_ones / den
#         ParameterArray[0, j] = 1 - ParameterArray[1, j]
#         if ((ParameterArray[1, j] == 0) or (ParameterArray[0, j] == 0)):
#             print('Divide by zero for j = ', j)
#             print(ParameterArray[0, j])
#             print(ParameterArray[1, j])
#     return ParameterArray
#
#
# def PredictProbabalitiesTestData(XTest, ParameterArray):
#     NumberOfTestingExamples = len(XTest)
#     NumberOfFeatures = len(XTest[0, :])
#     averageProbabilityOfDataSet = 0.0
#     for i in range(0, NumberOfTestingExamples):
#         probOfTestExample = 0.0
#         for j in range(0, NumberOfFeatures):
#             a = XTest[i, j]
#             probOfTestExample = probOfTestExample + math.log(ParameterArray[a, j], 2)
#         averageProbabilityOfDataSet = averageProbabilityOfDataSet + probOfTestExample
#     averageProbabilityOfDataSet = averageProbabilityOfDataSet / float(NumberOfTestingExamples)
#     print(" avg : ",averageProbabilityOfDataSet)
#     return
#
#
# x_tr = extractData(root_path + train_file + ".data")
# p = findingParameters(x_tr)
# x_te = extractData(root_path + test_file + ".data")
# PredictProbabalitiesTestData(x_te, p)

# -------------------------------------------------


p_array = np.zeros((2, n_features))
p_array[1] = (x_train.sum(axis=0) + 1) / (n_size + 2)
p_array[0] = 1 - p_array[1]
p_array = np.log2(p_array)
sum_ones = p_array[1].sum()
sum_zeros = p_array[0].sum()


log_likelihood = (x_test.dot(p_array[1]) + (sum_zeros - x_test.dot(p_array[0]))).sum()/len(x_test)
print("Log Likelihhod: ",log_likelihood)





end = timeit.default_timer()
print("Execution Time: ", end - start)
