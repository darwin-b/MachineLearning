import numpy as np
import pandas as pd


class node:
    def __init__(self,feature, branchList):
        self.feature = feature
        self.branches = branchList

    def travel(self):
        print(self.feature)
        for branch in self.branches:
            self.branches[branch].travel()
        print("level")



# Read Data
data1 = pd.DataFrame({"toothed":["True","True","True","False","True","True","True","True","True","False"],
                     "hair":["True","True","False","True","True","True","False","False","True","False"],
                     "breathes":["True","True","True","True","True","True","False","True","True","True"],
                     "legs":["True","True","False","True","True","True","False","False","True","True"],
                     "species":["Mammal","Mammal","Reptile","Mammal","Mammal","Mammal","Reptile","Reptile","Mammal","Reptile"]},
                    columns=["toothed","hair","breathes","legs","species"])

data2 = pd.DataFrame({"Outlook":["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
                     "Temp":["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
                     "Humidity":["High","High","High","High","Normal","Normal","Normal","High","Normal","Normal","Normal","High","Normal","High"],
                     "Wind":["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
                     "Play":["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"]},
                    columns=["Outlook","Temp","Humidity","Wind","Play"])

data = pd.read_csv("./Data/train_c300_d100.csv", header=None)


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
        viGain = viGain - partitionProbabilty*featureVi

        # print("Feature Entropy - ",featureValue,featureEntropy)

        # Drop the split feature from partitioned data
        dataPartition = dataPartition.drop(feature,axis=1)
        dataSplitList[featureValue]=dataPartition
        # print("hi",l[featureValue])

    # print("Gain: ", viGain)
    return viGain,dataSplitList

def viDTree(data,target):

    datasetVi = varianceImpurity(data,target)
    if datasetVi == 0:
        print("VI is 0 & No split required")
        for leaf in data[target]:
            value=leaf
            break
        return node(value,{})

    print("Dataset VI : ", datasetVi)

    leastViGain=100
    selectedFeature=None
    splitData=None
    for feature in data.columns:

        #CHeck to ignore target column
        if feature == data.columns[-1]:
            continue

        gain,split = viGain(data,feature,target)
        featureViGain = datasetVi - gain
        print("Feature :",feature," ViGain :",featureViGain)

        # Choosing split attribute
        if featureViGain<leastViGain:
            leastViGain = featureViGain
            selectedFeature=feature
            splitData = split

    print("Highest Variance Gain feature: ", selectedFeature)

    brlist={}

    print(" ")
    for split in splitData.keys():
        print("split Value : ",split)
        brlist[split] = viDTree(splitData[split],target)

    return node(selectedFeature,brlist)


target=500
feature=3

j=viDTree(data,target)
j.travel()



target1 = "species"
feature1 = "toothed"

target2 = "Play"
feature2 = "Outlook"