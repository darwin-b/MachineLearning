import sys
import os
import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize

nltk.download('punkt')

data_path = "C:\\Users\\darwi\\OneDrive - " \
            "The University of Texas at Dallas\\Acads\\Machine Learning\\Assignments\\MachineLearning\\Data"

cwd = os.getcwd()



data = []


def read(file_path):
    with open(file_path, encoding='cp437') as file:
        text = file.read()
    return text


# Reading Datasets
for dataset in os.listdir(data_path):
    train_path = data_path + os.path.sep + dataset + os.path.sep + "train" + os.path.sep
    test_path = data_path + os.path.sep + dataset + os.path.sep + "test" + os.path.sep

    train_text = {}
    # test_text = {}

    #  Text initialization
    train_text["ham"] = ""
    train_text["spam"] = ""
    # test_text["ham"] = ""
    # test_text["spam"] = ""

    for doc in os.listdir(train_path + "ham"):
        train_text["ham"] = train_text["ham"] + read(train_path + "ham" + os.path.sep + doc)

    for doc in os.listdir(train_path + "spam"):
        train_text["spam"] = train_text["spam"] + read(train_path + "spam" + os.path.sep + doc)

    # for doc in os.listdir(test_path + "ham"):
    #     test_text["ham"] = test_text["ham"] + read(test_path + "ham" + os.path.sep + doc)
    #
    # for doc in os.listdir(test_path + "spam"):
    #     test_text["spam"] = test_text["spam"] + read(test_path + "spam" + os.path.sep + doc)

    data.append(train_text)
    # data.append(test_text)


# WIP: Use below file to test processing text
# text1 = (read(data_path+os.path.sep+"enron1\\test"+os.path.sep+"spam"+os.path.sep+"4566.2005-05-24.GP.spam.txt"))
# text2 = (read(data_path+os.path.sep+"enron1\\test"+os.path.sep+"spam"+os.path.sep+"0046.2003-12-20.GP.spam.txt"))


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


# "Hello how are you. hello are good"

bag_train1_ham = bag_words(data[0]["ham"], {})
bag_train1_spam = bag_words(data[0]["spam"], {})

bag_train2_ham = bag_words(data[1]["ham"], {})
bag_train2_spam = bag_words(data[1]["spam"], {})

bag_train3_ham = bag_words(data[2]["ham"], {})
bag_train3_spam = bag_words(data[2]["spam"], {})


def conditional_prob(bag_ham, bag_spam, word, class_bag):
    bag_ham_length = 0  # sum of frequencies of words in ham
    bag_spam_length = 0  # sum of frequencies of words in Spam
    vocab_size = bag_ham.__len__()  # Vocabulary size/count

    for w in bag_ham:
        bag_ham_length += bag_ham[w]

    for w in bag_spam:
        bag_spam_length += bag_spam[w]
        if w not in bag_ham:
            vocab_size += 1

    if class_bag == "ham":
        if word in bag_ham:
            return (bag_ham[word] + 1) / (bag_ham_length + vocab_size)
        else:
            return 1 / (bag_ham_length + vocab_size)
    else:
        if word in bag_spam:
            return (bag_spam[word] + 1) / (bag_spam_length + vocab_size)
        else:
            return 1 / (bag_spam_length + vocab_size)


# bag1 = bag_words(" Chinese Beijing Chinese Chinese Chinese Shanghai Chinese Macao",{})
# bag2 = bag_words("Tokyo Japan Chinese",{})


#dp={}
# dp[("ham","chinese")]=conditional_prob(bag1,bag2,"chinese","ham")
#
# if ("ham","tokyo") in dp:
#     print("Hi")
#
# c1=conditional_prob(bag1,bag2,"chinese","ham")
# c2=conditional_prob(bag1,bag2,"chinese","spam")
# c3=conditional_prob(bag1,bag2,"japan","ham")
# c4=conditional_prob(bag1,bag2,"tokyo","spam")

# conditional_prob(bag_train1_ham,bag_train1_spam,"re","ham")


test_path_ham = data_path + os.path.sep + "hw2" + os.path.sep + "test" + os.path.sep + "ham" + os.path.sep
ham_count = os.listdir(test_path_ham).__len__()

test_path_spam = data_path + os.path.sep + "hw2" + os.path.sep + "test" + os.path.sep + "spam" + os.path.sep
spam_count = os.listdir(test_path_spam).__len__()

dp = {}
pred_prob_ham = []
for file in os.listdir(test_path_ham):
    bag_test_ham = bag_words(read(test_path_ham + file), {})
    prob = np.log10(ham_count / (ham_count + spam_count))
    for word in bag_test_ham:
        if ("ham", word) not in dp:
            dp[("ham", word)] = np.log10(conditional_prob(bag_train3_ham, bag_train3_spam, word, "ham"))
        cp = dp[("ham", word)]
        prob = prob + (cp * bag_test_ham[word])

    pred_prob_ham.append(prob)

dp2 = {}
pred_prob_spam = []
for file in os.listdir(test_path_ham):
    bag_test_spam = bag_words(read(test_path_ham + file), {})
    prob = np.log10(spam_count / (ham_count + spam_count))
    for word in bag_test_spam:
        if ("spam", word) not in dp2:
            dp2[("spam", word)] = np.log10(conditional_prob(bag_train3_spam, bag_train3_spam, word, "spam"))
        cp = dp2[("spam", word)]
        prob = prob + (cp * bag_test_spam[word])

    pred_prob_spam.append(prob)

# caluclate accuracy & prediction

acc=0
for x in range(len(pred_prob_ham)):
    if pred_prob_ham[x] > pred_prob_spam[x]:
        acc +=1
    else:
        print(x)


print("Acc : ",acc/len(pred_prob_ham)," Count : ",acc)
# -------------------------test_spam--------------------------
dp = {}
pred_prob_ham = []
for file in os.listdir(test_path_spam):
    bag_test_ham = bag_words(read(test_path_spam + file), {})
    prob = np.log10(ham_count / (ham_count + spam_count))
    for word in bag_test_ham:
        if ("ham", word) not in dp:
            dp[("ham", word)] = np.log10(conditional_prob(bag_train3_ham, bag_train3_spam, word, "ham"))
        cp = dp[("ham", word)]
        prob = prob + (cp * bag_test_ham[word])

    pred_prob_ham.append(prob)

dp2 = {}
pred_prob_spam = []
for file in os.listdir(test_path_spam):
    bag_test_spam = bag_words(read(test_path_spam + file), {})
    prob = np.log10(spam_count / (ham_count + spam_count))
    for word in bag_test_spam:
        if ("spam", word) not in dp2:
            dp2[("spam", word)] = np.log10(conditional_prob(bag_train3_spam, bag_train3_spam, word, "spam"))
        cp = dp2[("spam", word)]
        prob = prob + (cp * bag_test_spam[word])

    pred_prob_spam.append(prob)

# caluclate accuracy & prediction

acc=0
for x in range(len(pred_prob_ham)):
    if pred_prob_ham[x] > pred_prob_spam[x]:
        acc +=1
    else:
        print(x)


print("Acc : ",acc/len(pred_prob_ham)," Count : ",acc)
