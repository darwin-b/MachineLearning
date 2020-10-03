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

data = []

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
                bag[word] =  1

    return bag

    # for word in clean_text:


def conditional_prob(bag, word, count):

    word_inDoc=0
    for w in bag:
        if word in bag[w].keys():
            word_inDoc+=1
    return (word_inDoc+1)/(count+2)



def doc_probability(testFile_path,vocab,class_prob,bag_ham,bag_spam,class_label,count):

    dp = {}
    pred_prob = []
    for file in os.listdir(testFile_path):
        bag_test = bag_words(read(testFile_path+os.path.sep+ file), {})
        prob=class_prob

        if class_label=="ham":

            for v in vocab:
                if v in bag_test:
                    if (class_label, v) not in dp:
                        dp[(class_label, v)] = np.log10(conditional_prob(bag_ham, v, count))
                    cp1 = dp[(class_label, v)]
                    prob = prob + cp1
                else:
                    for f in bag_ham:
                        if v in bag_ham[f]:
                            if (class_label, v) not in dp:
                                dp[(class_label, v)] = np.log10(conditional_prob(bag_ham, v, count))
                            cp2 = 1- dp[(class_label, v)]
                            prob = prob + cp2
                        else:
                            if ("spam", v) not in dp:
                                dp[("spam", v)] = np.log10(conditional_prob(bag_spam, v, count))
                            cp2 = 1- dp[("spam", v)]
                            prob = prob + cp2
        else:
            for v in vocab:
                if v in bag_test:
                    if (class_label, v) not in dp:
                        dp[(class_label, v)] = np.log10(conditional_prob(bag_spam, v, count))
                    cp1 = dp[(class_label, v)]
                    prob = prob + cp1
                else:
                    for f in bag_spam:
                        if v in bag_spam[f]:
                            if (class_label, v) not in dp:
                                dp[(class_label, v)] = np.log10(conditional_prob(bag_spam, v, count))
                            cp2 = 1- dp[(class_label, v)]
                            prob = prob + cp2
                        else:
                            if ("ham", v) not in dp:
                                dp[("ham", v)] = np.log10(conditional_prob(bag_ham, v, count))
                            cp2 = 1- dp[("ham", v)]
                            prob = prob + cp2




        # else:
        #     for word in bag_test:
        #
        #         if (class_label, word) not in dp:
        #             dp[(class_label, word)] = np.log10(conditional_prob(bag_spam, word, count))
        #         cp1 = dp[(class_label, word)]
        #         prob = prob + cp1
        #
        #     for file in bag_spam:
        #         for w in bag_spam[file]:
        #             if w not in bag_test:
        #                 if (class_label, w) not in dp:
        #                     dp[(class_label, w)] = np.log10(conditional_prob(bag_spam, w, count))
        #                 cp1 = 1 - dp[(class_label, w)]
        #                 prob = prob + cp1
        #
        #     for file in bag_ham:
        #         for w in bag_ham[file]:
        #             if w not in bag_test:
        #                 if ("ham", w) not in dp:
        #                     dp[("ham", w)] = np.log10(conditional_prob(bag_ham, w, count))
        #                 cp1 = 1 - dp[("ham", w)]
        #                 prob = prob + cp1

        pred_prob.append(prob)

    return pred_prob,dp


def accuracy(true, false, class_label):
    count = 0
    mislabel = []
    for x in range(len(true)):
        if true[x] > false[x]:
            count += 1
        else:
            mislabel.append(x)

    acc = count / len(true)

    if class_label == "ham":
        print("ham is ham : ", count, "/", len(true))
    else:
        print("spam is spam : ", count, "/", len(true))
    print(" acc : ", acc)

    return acc, mislabel


# bag1 = bag_words(" Chinese Beijing Chinese Chinese Chinese Shanghai Chinese Macao",{})
# bag2 = bag_words("Tokyo Japan Chinese",{})

# c1=conditional_prob(bag1,bag2,"chinese","ham")
# c2=conditional_prob(bag1,bag2,"chinese","spam")
# c3=conditional_prob(bag1,bag2,"japan","ham")
# c4=conditional_prob(bag1,bag2,"tokyo","spam")

# conditional_prob(bag_train1_ham,bag_train1_spam,"re","ham")

# ------------------------get path of Dataset with train & test sets nested in Dataset folder  ----------------------#
data_path = "C:\\Users\\darwi\\OneDrive - " \
            "The University of Texas at Dallas\\Acads\\Machine Learning\\Assignments\\MachineLearning\\Data\\hw2"

test_path_ham = data_path + os.path.sep + "test" + os.path.sep + "ham" + os.path.sep
test_path_spam = data_path + os.path.sep + "test" + os.path.sep + "spam" + os.path.sep
train_path_ham = data_path + os.path.sep + "train" + os.path.sep + "ham" + os.path.sep
train_path_spam = data_path + os.path.sep + "train" + os.path.sep + "spam" + os.path.sep

# --------------make bag of words from training data-------------------------#
bag_ham={}
for file in os.listdir(train_path_ham):
    bag_ham[file]= bag_words(read(train_path_ham + file), {})

bag_spam = {}
for file in os.listdir(train_path_spam):
    bag_spam[file] = bag_words(read(train_path_spam + file), {})

vocabulary={}
for f in bag_ham:
    for w in bag_ham[f]:
        if w not in vocabulary:
            vocabulary[w]=1

for f in bag_spam:
    for w in bag_spam[f]:
        if w not in vocabulary:
            vocabulary[w]=1

ham_count = os.listdir(train_path_ham).__len__()
spam_count = os.listdir(train_path_spam).__len__()

test_ham_count = os.listdir(test_path_ham).__len__()
test_spam_count = os.listdir(test_path_spam).__len__()

prob_ham = np.log10(ham_count / (ham_count + spam_count))
prob_spam = np.log10(spam_count / (ham_count + spam_count))

# -----------------------------Results Matrix------------------------------------
ham_true,cp_hamtrue = doc_probability(test_path_ham, vocabulary , prob_ham, bag_ham, bag_spam, "ham",ham_count)
ham_false,cp_hamfalse = doc_probability(test_path_ham, vocabulary , prob_ham, bag_ham, bag_spam, "spam",spam_count)

spam_true,cp_spamtrue = doc_probability(test_path_spam, vocabulary , prob_spam, bag_ham, bag_spam, "spam",spam_count)
spam_false,cp_spamfalse = doc_probability(test_path_spam, vocabulary , prob_spam, bag_ham, bag_spam, "ham",ham_count)

acc_ham,mislabel_ham  = accuracy(ham_true,ham_false,"ham")
acc_spam,mislabel_spam = accuracy(spam_true,spam_false,"spam")

acc_total = (acc_ham*test_ham_count+acc_spam*test_spam_count)/(test_ham_count+test_spam_count)
print("Total accuracy : ",acc_total)

file_name="resultsDNBMatrix_"+data_path.split(os.path.sep)[-1]+".txt"
with open(file_name,'w') as file:
    text = "ham is ham : "+str(acc_ham)+"\n"+"spam is spam : "+str(acc_spam)+"\n"+"Total accuracy : "+str(acc_total)+"\n"
    text = text+" mislabel ham file indices : "+repr(mislabel_ham)+"\n"+" mislabel spam file indices : "+repr(mislabel_spam)+"\n\n"
    text = text +"Bag of Ham :\n"+ repr(bag_ham)+"\n\n"+"Bag of Spam :\n"+ repr(bag_spam)
    file.write(text)