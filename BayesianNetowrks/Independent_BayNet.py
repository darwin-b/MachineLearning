import csv
import math
import sys
import numpy as np
import timeit

start = timeit.default_timer()

root_path = sys.argv[1]+"\\"

# root_path = "C:\\Users\\darwi\\OneDrive - The University of Texas at Dallas\\Acads\\Machine Learning\\Assignments\\MachineLearning\\Data\\small-10-datasets\\"
# train_file = "dna.ts"
# test_file = "dna.test"
# extension = ".data"

# datasets=["accidents","baudio","bnetflix","dna","jester","kdd","msnbc","nltcs","plants"]

datasets=["plants","nltcs","msnbc","kdd","jester","dna","bnetflix","baudio","accidents"]
# datasets=["dna"]

for dataset in datasets:


    print("------------------------- ",dataset," -------------------------------")
    x_train = np.loadtxt(root_path + dataset + ".ts.data", delimiter=',', dtype=int)
    x_test = np.loadtxt(root_path + dataset + ".test.data", delimiter=',', dtype=int)

    n_size = x_train.shape[0]
    n_features = x_train.shape[1]
    print("Size : ", n_size, " X ", n_features)

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
