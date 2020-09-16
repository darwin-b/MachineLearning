import numpy as np
import pandas as pd
import time
import copy

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



def varianceImpurity(data,target):
    targetColumn = data.loc[:, target].value_counts()

    # Calculate entropy of data set
    dataSize = data.shape[0]
    vi = 1
    for target in targetColumn:
        vi=vi*target/dataSize

    if vi==1:
        return 0

    return vi



def viGain(data,feature, target):

    viGain =0
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
        featureVi = varianceImpurity(dataPartition,target)
        viGain = viGain + partitionProbabilty*featureVi
        # print("VI Gain :",viGain)

        # print("Feature Entropy - ",featureValue,featureEntropy)

        # Drop the split feature from partitioned data
        dataPartition = dataPartition.drop(feature,axis=1)
        dataSplitList[featureValue]=dataPartition
        # print("hi",l[featureValue])

    # print("Gain: ", viGain)
    return viGain,dataSplitList

def viDTree(data,depth,target):

    datasetVi = varianceImpurity(data,target)
    if datasetVi == 0:
        # print("VI is 0 & No split required")
        for leaf in data[target]:
            value=leaf
            break
        return node(value,depth,-1,-1,{})

    # print("Dataset VI : ", datasetVi)

    highestViGain=-1
    selectedFeature=None
    splitData=None
    for feature in data.columns:

        #CHeck to ignore target column
        if feature == data.columns[-1]:
            continue

        gain,split = viGain(data,feature,target)
        featureViGain = datasetVi - gain
        # print("Feature :",feature," ViGain :",featureViGain," Gain: ",gain)

        # Choosing split attribute
        if featureViGain>highestViGain:
            highestViGain = featureViGain
            selectedFeature=feature
            splitData = split

    # print("Highest Variance Gain feature: ", selectedFeature)

    brlist={}

    # print(" ")
    for split in splitData.keys():
        # print("split Value : ",split)
        brlist[split] = viDTree(splitData[split],depth+1,target)

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
    # print("Accuracy : ",fileAccuracy[file]," Runtime : ",fileRunTime[file])


testDataFiles  = ["test_c300_d100","test_c300_d1000","test_c300_d5000","test_c500_d100","test_c500_d1000","test_c500_d5000","test_c1000_d100","test_c1000_d1000","test_c1000_d5000","test_c1500_d100","test_c1500_d1000","test_c1500_d5000","test_c1800_d100","test_c1800_d1000","test_c1800_d5000"]
validDataFiles = ["valid_c300_d100","valid_c300_d1000","valid_c300_d5000","valid_c500_d100","valid_c500_d1000","valid_c500_d5000","valid_c1000_d100","valid_c1000_d1000","valid_c1000_d5000","valid_c1500_d100","valid_c1500_d1000","valid_c1500_d5000","valid_c1800_d100","valid_c1800_d1000","valid_c1800_d5000"]
trainDataFiles = ["train_c300_d100","train_c300_d1000","train_c300_d5000","train_c500_d100","train_c500_d1000","train_c500_d5000","train_c1000_d100","train_c1000_d1000","train_c1000_d5000","train_c1500_d100","train_c1500_d1000","train_c1500_d5000","train_c1800_d100","train_c1800_d1000","train_c1800_d5000"]


# testDataFiles  = [testDataFiles[12]]
# validDataFiles = [validDataFiles[12]]
# trainDataFiles = [trainDataFiles[12]]
#

# Count number of files
files = testDataFiles.__len__()


filePred = {}
fileModel = {}
fileAccuracy = {}
fileTruePos = {}
fileRunTime={}

# temp = dTree(data2,0,"Play")

# DTree resultsMatrix with Variance as impurity heuristic
for file in range(0,files):

    start = time.time()
    x_train,y_train,dataTrain = data(trainDataFiles[file])
    x_test,y_test,dataTest = data(testDataFiles[file])

    # fileSize=dataTrain.shape[0]

    target = dataTrain.columns[-1]
    fileModel[file]=viDTree(dataTrain,0,target)

    # Calculate run time
    end = time.time()
    fileRunTime[file]= end-start

    fileAccuracy[file], filePred[file], fileTruePos[file] = accuracyMatrix(x_test,y_test,fileModel[file])
    print(testDataFiles[file]," ==> ","Accuracy : ",fileAccuracy[file]," Runtime : ",fileRunTime[file])

#
# files = testDataFiles.__len__()
# DTree resultsMatrix with Variance as impurity heuristic and depth based pruning
print("")
print("")
print("Depth based pruning with Variance as heuristic")
for file in range(0,files):
    # not yet working/ using valid data files : to be used in reduced error pruning
    x_valid,y_valid,dataValid = data(validDataFiles[file])
    x_test,y_test,dataTest = data(testDataFiles[file])
    print(testDataFiles[file])

    # Depth --> 20,15,10,5
    for depth in range(20,0,-5):
        tuningModel=copy.deepcopy(fileModel[file])
        tuningModel.depthPrune(depth)

        acc,pre,pos=accuracyMatrix(x_test,y_test,tuningModel)
        print("depth :",depth,"accuracy :",acc)


f = open( 'Models-VarGain.txt', 'w' )
f.write( 'dict = ' + repr(fileModel) + '\n' )
f.close()

f = open( 'Pred-VarGain.txt', 'w' )
f.write( 'dict = ' + repr(filePred) + '\n' )
f.close()

f = open( 'Accuracy-VarGain.txt', 'w' )
f.write( 'dict = ' + repr(fileAccuracy) + '\n' )
f.close()

f = open( 'Runtimes-VarGain.txt', 'w' )
f.write( 'dict = ' + repr(fileRunTime) + '\n' )
f.close()

f = open( 'TruePositives-VarGain.txt', 'w' )
f.write( 'dict = ' + repr(fileTruePos) + '\n' )
f.close()

