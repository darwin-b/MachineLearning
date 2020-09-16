import numpy as np
import pandas as pd
import time
import copy

# from joblib import Parallel, delayed
# import multiprocessing


class node:
    def __init__(self,feature,depth,commonClass,splitSize,branchList):
        self.feature = feature
        self.depth = depth
        self.commonClass = commonClass
        self.splitSize = splitSize

        self.branches = branchList

    def travel(self):
        print(self.feature)
        for branch in self.branches:
            self.branches[branch].travel()
        # print("level")

    def depthPrune(self,maxLimit):
        if self.depth==maxLimit:
            self.feature=self.commonClass
            self.branches={}
        for branch in self.branches:
            self.branches[branch].depthPrune(maxLimit)



def entropy(data, target):
    # Get frequency of target classes in target column in pd frame
    targetColumn = data.loc[:, target].value_counts()

    # Calculate entropy of data set
    dataSize = data.shape[0]
    entropy = 0
    for target in targetColumn:
        p = target / dataSize
        entropy = -p * np.log2(p) + entropy

    # print(entropy)
    return entropy

def Gain(data, feature, target):

    # initialising feature entropy & attriubte gain
    featureEntropy = 0
    gain =0
    dataSplitList = {}

    # groupData = data.groupby([feature,target])
    # Group Data based on feature
    groupData = data.groupby([feature])
    featureValueSize = groupData.size()

    # Determining Data size
    dataSize = data.shape[0]

    # Partiton/split data based on feature values
    for featureValue in featureValueSize.keys():

        partitionSize = featureValueSize[featureValue]
        # print("partiton size : ",partitionSize)
        partitionProbabilty = partitionSize/dataSize
        # print("Partition probabilty for featurevalue :",featureValue,partitionProbabilty)
        # Partioning the data
        dataPartition = data[data[feature]==featureValue]
        # print(dataPartition)

        # Determine individual featurevalue entropy
        featureEntropy = entropy(dataPartition,target)
        gain = partitionProbabilty*featureEntropy + gain

        # print("Feature Entropy - ",featureValue,featureEntropy)

        # Drop the split feature from partitioned data
        dataPartition = dataPartition.drop(feature,axis=1)
        dataSplitList[featureValue]=dataPartition
        # print("hi",l[featureValue])

    # print("Gain: ", gain)
    return gain,dataSplitList

def dTree(data,depth,target):

    datasetEntropy = entropy(data,target)
    if datasetEntropy == 0:
        # print("Entropy is 0 & No split required")
        for leaf in data[target]:
            value=leaf
            break
        return node(value,depth,-1,-1,{})

    # print("Dataset Entropy : ", datasetEntropy)

    highestInfoGain=-1
    selectedFeature=None
    splitData=None
    for feature in data.columns:

        #CHeck to ignore target column
        if feature == data.columns[-1]:
            continue

        gain,split = Gain(data,feature,target)
        featureInfoGain = datasetEntropy - gain
        # print("Feature :",feature," InfoGain :",featureInfoGain)

        # Choosing split attribute
        if featureInfoGain>highestInfoGain:
            highestInfoGain = featureInfoGain
            selectedFeature=feature
            splitData = split

    # print("Highest Info Gain feature: ", selectedFeature)

    brlist={}

    # print(" ")
    for split in splitData.keys():
        # print("split Value : ",split)
        brlist[split] = dTree(splitData[split],depth+1,target)

    splitSize=data.loc[:, target].value_counts()
    commonClass= splitSize.idxmax()

    return node(selectedFeature,depth,commonClass,splitSize,brlist)


def pred(x,root):
    if len(root.branches)==0:
        # print(root.feature)
        return root.feature

    value=x[root.feature]
    # print(root.feature, value)
    return pred(x,root.branches[value])

def data(fileName):
    relativePath ="./Data/"
    filePath=relativePath+fileName+".csv"
    data = pd.read_csv(filePath, header=None)

    dataRows = data.shape[0]
    dataCols = data.shape[1]

    # print(dataRows,dataCols)

    y=data[data.columns[-1]]
    x=data[data.columns[0:dataCols-1]]

    return x,y,data

def accuracyMatrix(x_val,y_val,model):
    y_pred = {}
    dataSize = x_val.shape[0]
    for rowIndex in x_val.index:
        row = x_val.iloc[rowIndex, :]
        y_pred[rowIndex] = pred(row, model)
    # filePred[file]=y_pred

    count = 0
    truePos = []
    for each in y_test.keys():
        # print(each)
        if y_test[each] == y_pred[each]:
            truePos.append(each)
            count = count + 1
    # fileTruePos[file]=truePos
    accuracy=count/dataSize
    # fileAccuracy[file]=count/fileSize

    return accuracy,y_pred,truePos
    print("Accuracy : ",fileAccuracy[file]," Runtime : ",fileRunTime[file])

# Note: for feature "outlook" & value "outcast" the log values are calculated correctly despite of invalid of log(0) value

testDataFiles  = ["test_c300_d100","test_c300_d1000","test_c300_d5000","test_c500_d100","test_c500_d1000","test_c500_d5000","test_c1000_d100","test_c1000_d1000","test_c1000_d5000","test_c1500_d100","test_c1500_d1000","test_c1500_d5000","test_c1800_d100","test_c1800_d1000","test_c1800_d5000"]
validDataFiles = ["valid_c300_d100","valid_c300_d1000","valid_c300_d5000","valid_c500_d100","valid_c500_d1000","valid_c500_d5000","valid_c1000_d100","valid_c1000_d1000","valid_c1000_d5000","valid_c1500_d100","valid_c1500_d1000","valid_c1500_d5000","valid_c1800_d100","valid_c1800_d1000","valid_c1800_d5000"]
trainDataFiles = ["train_c300_d100","train_c300_d1000","train_c300_d5000","train_c500_d100","train_c500_d1000","train_c500_d5000","train_c1000_d100","train_c1000_d1000","train_c1000_d5000","train_c1500_d100","train_c1500_d1000","train_c1500_d5000","train_c1800_d100","train_c1800_d1000","train_c1800_d5000"]

testDataFiles  = [testDataFiles[6],testDataFiles[12]]
validDataFiles = [validDataFiles[6],validDataFiles[12]]
trainDataFiles = [trainDataFiles[6],trainDataFiles[12]]

# testDataFiles  = ["test_c300_d100","test_c300_d1000"]
# validDataFiles = ["valid_c300_d100","valid_c300_d1000"]
# trainDataFiles = ["train_c300_d100","train_c300_d1000"]

# Count number of files
files = testDataFiles.__len__()


filePred = {}
fileModel = {}
fileAccuracy = {}
fileTruePos = {}
fileRunTime={}

# temp = dTree(data2,0,"Play")

# DTree resultsMatrix with Entropy as impurity heuristic
for file in range(0,files):

    start = time.time()
    x_train,y_train,dataTrain = data(trainDataFiles[file])
    x_test,y_test,dataTest = data(testDataFiles[file])

    # fileSize=dataTrain.shape[0]

    target = dataTrain.columns[-1]
    fileModel[file]=dTree(dataTrain,0,target)

    # Calculate run time
    end = time.time()
    fileRunTime[file]= end-start

    fileAccuracy[file], filePred[file], fileTruePos[file] = accuracyMatrix(x_test,y_test,fileModel[file])
    print(testDataFiles[file]," ==> ","Accuracy : ",fileAccuracy[file]," Runtime : ",fileRunTime[file])



# DTree resultsMatrix with Entropy as impurity heuristic and depth based pruning
print("")
print("")
print("Depth based pruning with Entropy as heuristic")
for file in range(0,files):
    # not yet working/ using valid data files : to be used in reduced error pruning
    x_valid,y_valid,dataValid = data(validDataFiles[file])
    x_test,y_test,dataTest = data(testDataFiles[file])
    print(testDataFiles[file])
    for depth in range(20,0,-5):
        tuningModel=copy.deepcopy(fileModel[file])
        tuningModel.depthPrune(depth)

        acc,pre,pos=accuracyMatrix(x_test,y_test,tuningModel)
        print("depth :",depth,"accuracy :",acc)


# dep =5
# tuningModel=copy.deepcopy(fileModel[0])
# tuningModel.depthPrune(dep)
#
# acc, pre, pos = accuracyMatrix(x_test, y_test, fileModel[0])
# print("depth :", dep,"accuracy :", acc)
#
# acc, pre, pos = accuracyMatrix(x_test, y_test, tuningModel)
# print("depth :", dep,"accuracy :", acc)

# f = open( 'Models-InfoGain.txt', 'w' )
# f.write( 'dict = ' + repr(fileModel) + '\n' )
# f.close()
#
# f = open( 'Pred-InfoGain.txt', 'w' )
# f.write( 'dict = ' + repr(filePred) + '\n' )
# f.close()
#
# f = open( 'Accuracy-InfoGain.txt', 'w' )
# f.write( 'dict = ' + repr(fileAccuracy) + '\n' )
# f.close()
#
# f = open( 'Runtimes-InfoGain.txt', 'w' )
# f.write( 'dict = ' + repr(fileRunTime) + '\n' )
# f.close()
#
# f = open( 'TruePositives-InfoGain.txt', 'w' )
# f.write( 'dict = ' + repr(fileTruePos) + '\n' )
# f.close()
#
#
# runtime=0
# for each in fileRunTime:
#     runtime = fileRunTime[each] +runtime
# print("Total Runtime",runtime)
#
#
#
# # num_cores = multiprocessing.cpu_count()
# # print("Cores available : ",num_cores)
#
# # Parallel(n_jobs=num_cores)(delayed(crawl.get_episode)(title) for title in episodes_list)
