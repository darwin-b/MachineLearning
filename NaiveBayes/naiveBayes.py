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
            if word in bag.keys():
                bag[word] = bag[word] + 1
            else:
                bag[word] = 1
    return bag

    # for word in clean_text:


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


def doc_probability(testFile_path,class_prob,bag_ham,bag_spam,class_label):

    # ham_count = os.listdir(testFile_path).__len__()
    # spam_count = os.listdir(testFile_path_spam).__len__()
    #
    # if class_label=="ham":
    #     prob = np.log10(ham_count / (ham_count + spam_count))
    # else:
    #     prob = np.log10(spam_count / (ham_count + spam_count))

    dp = {}
    pred_prob = []
    for file in os.listdir(testFile_path):
        bag_test = bag_words(read(testFile_path+os.path.sep+ file), {})
        prob=class_prob
        for word in bag_test:
            if (class_label, word) not in dp:
                dp[(class_label, word)] = np.log10(conditional_prob(bag_ham, bag_spam, word, class_label))
            cp = dp[(class_label, word)]
            prob = prob + (bag_test[word]*cp)

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

    if (class_label == "ham"):
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
            "The University of Texas at Dallas\\Acads\\Machine Learning\\Assignments\\MachineLearning\\Data\\enron4"

test_path_ham = data_path + os.path.sep + "test" + os.path.sep + "ham" + os.path.sep
test_path_spam = data_path + os.path.sep + "test" + os.path.sep + "spam" + os.path.sep
train_path_ham = data_path + os.path.sep + "train" + os.path.sep + "ham" + os.path.sep
train_path_spam = data_path + os.path.sep + "train" + os.path.sep + "spam" + os.path.sep

# --------------make bag of words from training data-------------------------#
bag_ham={}
for file in os.listdir(train_path_ham):
    bag_ham= bag_words(read(train_path_ham + file),bag_ham)

bag_spam = {}
for file in os.listdir(train_path_spam):
    bag_spam = bag_words(read(train_path_spam + file), bag_spam)

ham_count = os.listdir(test_path_ham).__len__()
spam_count = os.listdir(test_path_spam).__len__()


prob_ham = np.log10(ham_count / (ham_count + spam_count))
prob_spam = np.log10(spam_count / (ham_count + spam_count))

# -----------------------------Results Matrix------------------------------------
ham_true,cp_hamtrue = doc_probability(test_path_ham, prob_ham, bag_ham, bag_spam, "ham")
ham_false,cp_hamfalse = doc_probability(test_path_ham, prob_ham, bag_ham, bag_spam, "spam")

spam_true,cp_spamtrue = doc_probability(test_path_spam, prob_spam, bag_ham, bag_spam, "spam")
spam_false,cp_spamfalse = doc_probability(test_path_spam, prob_spam, bag_ham, bag_spam, "ham")

acc_ham,mislabel_ham  = accuracy(ham_true,ham_false,"ham")
acc_spam,mislabel_spam = accuracy(spam_true,spam_false,"spam")

acc_total = (acc_ham*ham_count+acc_spam*spam_count)/(ham_count+spam_count)
print("Total accuracy : ",acc_total)

file_name="resultsMatrix_"+data_path.split(os.path.sep)[-1]+".txt"
with open(file_name,'w') as file:
    text = "ham is ham : "+str(acc_ham)+"\n"+"spam is spam : "+str(acc_spam)+"\n"+"Total accuracy : "+str(acc_total)+"\n"
    text = text+" mislabel ham file indices : "+repr(mislabel_ham)+"\n"+" mislabel spam file indices : "+repr(mislabel_spam)+"\n\n"
    text = text +"Bag of Ham :\n"+ repr(bag_ham)+"\n\n"+"Bag of Spam :\n"+ repr(bag_spam)
    file.write(text)