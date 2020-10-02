import sys
import os
import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
import time
import random

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

def sigmoid2(x):
    return np.exp(-1 * x) / (1 + np.exp(-1 * x))



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

# -------------------- Splitting Data : 70-30 Ratio------------------------- #
np.random.shuffle(data_X)
splitValue= int((hamFiles_count+spamFiles_count)*0.7)
train_X,valid_X = data_X[:splitValue,:], data_X[splitValue:,:]
train_y,valid_y = data_X[:splitValue,-1], data_X[splitValue:,-1]

# -----------------Data engineering done-------------------------------------#

# with open("baggedData.txt",'w') as fileWriter:
#     fileWriter.write(repr(data_X))
# note : dimension of train_x is count_features+1. [last column is y values]
print("------------------------Data Engineering done------------------------")
weights = np.zeros(count_features)
learning_rate = 0.5
l_lambda = 0.1

# llbm = [x for x in range(1,0.5)]

for iterations in range(200):
    weighted_features = weights*train_X[:,:-1]
    linear_score =np.sum(weighted_features,axis=1)
    diff_matrix=train_y-sigmoid(linear_score)
    errorWeighted_features= np.multiply(diff_matrix,np.transpose(train_X[:,:-1]))
    weights = weights + learning_rate*np.sum(errorWeighted_features,axis=1) - learning_rate*l_lambda*weights



#---------validation-----------#
valid_weighted_features = weights*valid_X[:,:-1]
valid_linear_score =np.sum(valid_weighted_features,axis=1)
valid_ham_predict=sigmoid(valid_linear_score)

count=0
count2=0
true_ham=0
true_spam=0
misLabel=[]
for each in range(len(valid_y)):
    # if valid_ham_predict[each]>0.5:
    #     valid_ham_predict[each]=1
    # else:
    #     valid_ham_predict[each]=0
    # if valid_y[each]== valid_ham_predict[each]:
    #     count +=1
    if valid_y[each]==1:
        true_ham+=1
    else:
        true_spam+=1

    if valid_y[each]==1 and valid_ham_predict[each]>0.5:
        count+=1
    if valid_y[each]==0 and valid_ham_predict[each]<0.5:
        count2+=1

print("Valid ham is ham : ", count," Acc : ",count/true_ham," true ham: ",true_ham)
print("Valid spam is spam : ", count2," Acc : ",count2/true_spam," true spam: ",true_spam)




# -------------test samples---------------------------#
testHam_files_count=os.listdir(test_path_ham).__len__()
testSpam_files_count=os.listdir(test_path_spam).__len__()
test_ham=np.zeros((testHam_files_count,count_features+1))
test_spam=np.zeros((testSpam_files_count,count_features+1))


index_file=0
for file in os.listdir(test_path_ham):
    words = bag_words(read(test_path_ham + file), {})
    for word in words:
        if word in baggedIndex:
            test_ham[index_file][baggedIndex[word]] = words[word]

    index_file += 1

# ----------------pedict ham----------------------


testHam_weighted_features = weights*test_ham[:,:-1]
testHam_linear_score =np.sum(testHam_weighted_features,axis=1)
test_ham_predict=sigmoid(testHam_linear_score)

count=0
count2=0
true_ham=len(test_ham_predict)
true_spam=0
misLabel=[]
for each in range(len(test_ham_predict)):
    # if valid_ham_predict[each]>0.5:
    #     valid_ham_predict[each]=1
    # else:
    #     valid_ham_predict[each]=0
    # if valid_y[each]== valid_ham_predict[each]:
    #     count +=1
    if test_ham_predict[each]>0.5:
        count+=1
    else:
        count2+=1

print("Valid ham is ham : ", count," Acc : ",count/true_ham," true ham: ",true_ham)
# print("Valid ham is ham : ", count," Acc : ",count/true_spam," true ham: ",true_spam)



# ---------test spam-------------------------------------------------------#
index_file=0
for file in os.listdir(test_path_spam):
    words = bag_words(read(test_path_spam + file), {})
    for word in words:
        if word in baggedIndex:
            test_spam[index_file][baggedIndex[word]] = words[word]

    index_file += 1

testSpam_weighted_features = weights*test_spam[:,:-1]
testSpam_linear_score =np.sum(testSpam_weighted_features,axis=1)
test_spam_predict=sigmoid(testSpam_linear_score)

count=0
count2=0
true_ham=0
true_spam=len(test_spam_predict)
misLabel=[]
for each in range(len(test_spam_predict)):
    # if valid_ham_predict[each]>0.5:
    #     valid_ham_predict[each]=1
    # else:
    #     valid_ham_predict[each]=0
    # if valid_y[each]== valid_ham_predict[each]:
    #     count +=1
    if test_spam_predict[each]>0.5:
        count+=1
    else:
        count2+=1

print("Valid spam is spam : ", count," Acc : ",count/true_spam," true spam: ",true_spam)
# print("Valid ham is ham : ", count," Acc : ",count/true_spam," true ham: ",true_spam)

