import sys
import os
import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
import time


nltk.download('punkt')
data_path = "C:\\Users\\darwi\\OneDrive - " \
            "The University of Texas at Dallas\\Acads\\Machine Learning\\Assignments\\MachineLearning\\Data"
cwd = os.getcwd()


def read(file_path):
    with open(file_path, encoding='cp437') as file:
        text = file.read()
    return text


def bag_words(text_data, bag):
    clean_text = nltk.sent_tokenize(text_data)
    for i in range(len(clean_text)):
        clean_text[i] = re.sub(r'\d', ' ', clean_text[i])  # Matches digits and replaces with blank space
        clean_text[i] = re.sub(r'\W', ' ', clean_text[i])  # Matches non-word and replaces with blank space
        clean_text[i] = re.sub(r'\s+', ' ', clean_text[i])  # Matches white-space and replaces with blank space
        clean_text[i] = clean_text[i].lower()  # Converts text to lower-case

    for sentence in clean_text:
        words = nltk.word_tokenize(sentence)
        for word in words:
            if word in bag.keys():
                bag[word] = bag[word] + 1
            else:
                bag[word] = 1
    return bag

    # for word in clean_text:


def sigmoid(x):
    return 1/(1+np.exp(-1*x))



data_path = "C:\\Users\\darwi\\OneDrive - " \
            "The University of Texas at Dallas\\Acads\\Machine Learning\\Assignments\\MachineLearning\\Data\\hw2"

test_path_ham = data_path + os.path.sep + "test" + os.path.sep + "ham" + os.path.sep
test_path_spam = data_path + os.path.sep + "test" + os.path.sep + "spam" + os.path.sep
train_path_ham = data_path + os.path.sep + "train" + os.path.sep + "ham" + os.path.sep
train_path_spam = data_path + os.path.sep + "train" + os.path.sep + "spam" + os.path.sep

bag={}
for file in os.listdir(train_path_ham):
    bag= bag_words(read(train_path_ham + file),bag)

# bag_spam = {}
for file in os.listdir(train_path_spam):
    bag = bag_words(read(train_path_spam + file), bag)

hamFiles_count = os.listdir(train_path_ham).__len__()
spamFiles_count = os.listdir(train_path_spam).__len__()

data_X = np.zeros((hamFiles_count+spamFiles_count,bag.__len__()))
data_y = np.ones((hamFiles_count+spamFiles_count,1))
data_y[hamFiles_count:,0]=0

z= os.listdir(test_path_ham)

baggedIndex={}
index=0
index_file=0
for file in os.listdir(train_path_ham):
    words = bag_words(read(train_path_ham + file),{})
    for word in words:
        if word not in baggedIndex:
            baggedIndex[word]=index
            data_X[index_file][index]=words[word]
            index +=1
        else:
            data_X[index_file][baggedIndex[word]]=words[word]

    index_file +=1

for file in os.listdir(train_path_spam):
    words = bag_words(read(train_path_spam + file),{})
    for word in words:
        if word not in baggedIndex:
            baggedIndex[word]=index
            data_X[index_file][index]=words[word]
            index +=1
        else:
            data_X[index_file][baggedIndex[word]]=words[word]

    index_file +=1

# -----------------Data engineering done-------------------------------------#



j=sigmoid(np.array([1,2,3]))
print(j)