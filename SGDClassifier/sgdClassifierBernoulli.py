import sys
import os
import pandas as pd
import numpy as np
import nltk
import re
from sklearn import linear_model
from nltk.tokenize import word_tokenize
import time
import random

nltk.download('punkt')
# data_path = "C:\\Users\\darwi\\OneDrive - " \
#             "The University of Texas at Dallas\\Acads\\Machine Learning\\Assignments\\MachineLearning\\Data"
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
            if word not in bag.keys():
                bag[word] = 1

    return bag


def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def sigmoid2(x):
    return np.exp(-1 * x) / (1 + np.exp(-1 * x))



# data_path = "C:\\Users\\darwi\\OneDrive - " \
#             "The University of Texas at Dallas\\Acads\\Machine Learning\\Assignments\\MachineLearning\\Data\\enron1"

data_path=sys.argv[1]
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

count_features = bag.__len__()

hamFiles_count = os.listdir(train_path_ham).__len__()
spamFiles_count = os.listdir(train_path_spam).__len__()

data_X = np.zeros((hamFiles_count+spamFiles_count,count_features+1))
data_X[0:hamFiles_count,-1]=1
data_X[hamFiles_count:,-1]=0

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

# ----------------------------- Splitting Data : 70-30 Ratio------------------------- #
np.random.shuffle(data_X)
splitValue= int((hamFiles_count+spamFiles_count)*0.7)
train_X,valid_X = data_X[:splitValue,:-1], data_X[splitValue:,:-1]
train_y,valid_y = data_X[:splitValue,-1], data_X[splitValue:,-1]


# default penalty --> L2 regularisation , tol-> (epsilon) for which the iterations stops before max iterations reached
SGDClf = linear_model.SGDClassifier(max_iter = 1000, tol=1e-3)
SGDClf.fit(train_X, train_y)

pred=SGDClf.predict(valid_X)

count=0
for x in range(len(valid_y)):
    if valid_y[x]==pred[x]:
        count+=1
print("--------------------------validation Results--------------------------")
print("Accuracy : ",count/len(valid_y))


# ------------------------ Read Test Data set-----------------------------#

testHam_files_count=os.listdir(test_path_ham).__len__()
testSpam_files_count=os.listdir(test_path_spam).__len__()
test_ham=np.zeros((testHam_files_count,count_features))
test_spam=np.zeros((testSpam_files_count,count_features))

# ----------------------------------Predict test ham--------------------------------------------#
index_file=0
for file in os.listdir(test_path_ham):
    words = bag_words(read(test_path_ham + file), {})
    for word in words:
        if word in baggedIndex:
            test_ham[index_file][baggedIndex[word]] = words[word]

    index_file += 1

pred_ham = SGDClf.predict(test_ham)

count1=0
for x in range(len(test_ham)):
    if pred_ham[x]==1:
        count1+=1
print("\n--------------------------Test Dataset--------------------------")
print("--------------------------ham is ham --------------------------")
print("Accuracy : ",count1/testHam_files_count)

# ----------------------------------Predict test spam--------------------------------------------#

index_file=0
for file in os.listdir(test_path_spam):
    words = bag_words(read(test_path_spam + file), {})
    for word in words:
        if word in baggedIndex:
            test_spam[index_file][baggedIndex[word]] = words[word]

    index_file += 1

pred_spam = SGDClf.predict(test_spam)
count2=0
for x in range(len(test_spam)):
    if pred_spam[x]==0:
        count2+=1

print("--------------------------spam is spam --------------------------")
print("Accuracy : ",count2/testSpam_files_count)

tp = count1
tn = count2
fp = testHam_files_count - count1
fn = testSpam_files_count - count2

acc=(tp+tn)/(tp+tn+fp+fn)
precision=(tp)/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*(recall * precision) / (recall + precision)

print("\n Total Accuracy on test files : ",acc)
print(" precision : ",precision)
print(" Recall : ",recall)
print(" F1_score : ",f1_score)

file_name="resultsSGDBernoulli_"+data_path.split(os.path.sep)[-1]+".txt"
with open(file_name,'w') as file:
    text = "Trained with shuffled 70-30 Data split into training & validation Data\n\n"
    text = text + "--------------Validation Results------------------" + "\n\n"
    text = text + "validation Accuracy : " + repr(count/len(valid_y)) + "\n\n\n"

    text = text + "--------------Results Test Data------------------"+"\n"
    text = text + "\n Accuracy on test files : "+ str(acc) + "\n"
    text = text + " precision : " + str(precision) + "\n"
    text = text + " Recall : " + str(recall) + "\n"
    text = text + " F1_score : " + str(f1_score) + "\n"
    file.write(text)