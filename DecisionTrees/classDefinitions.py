import numpy as np
import pandas as pd


class node:
    def __init__(self, feature, branchList):
        self.feature = feature
        self.branches = branchList

    def travel(self):
        print(self.feature)
        for branch in self.branches:
            self.branches[branch].travel()


tree = node("outlook", [node("humidity", [node("yes", []), node("no", [])]), node("overcast", [node("yes", [])]),
                        node("wind", [node("yes", []), node("no", [])])])

tree = node("outlook", {
    "sunny":node("humidity",{
        "high":{
            "no":{}
        },
        "normal":{
            "yes":{}
        }
    }),
    "overcast":node("Yes",{}),
    "rain": node("wind",{
        "strong":{
            "no":{}
        },
        "weak":{
            "yes":{}
        }
    })
})


# def computeGini(feature,data):
#     df =


def growTree():
    x = 1


test300_1000 = pd.read_csv("./Data/test_c300_d1000.csv", header=None)
y300_1000 = test300_1000.iloc[:,500:501]
x300_1000 = test300_1000.iloc[:,0:500]

featureLabel = 2
y_resultLabel = 500
positiveLabel = [int(1)]
df = test300_1000.loc[:,[featureLabel,y_resultLabel]]
dfGrouped=df.groupby(featureLabel)

# Method 1
feature_values = dfGrouped.groups.keys()
for value in feature_values:
    j=df.loc[df[featureLabel]==value]
    positive=j

# Method 2
for distinctFeatureValue in dfGrouped.groups:
    # print(group)
    rows_featureValue = df.loc[dfGrouped.groups[distinctFeatureValue], :]
    y_resultGrouped = rows_featureValue.groupby()


temp = {
    "a":1,
    "b":2,
    "c":3,
}

for t2 in temp:
    print(t2)

print(len(temp))

tree = node("outlook", {
    "sunny":node("humidity",{
        "high":{
            "no":{}
        },
        "normal":{
            "yes":{}
        }
    }),
    "overcast":node("Yes",{}),
    "rain": node("wind",{
        "strong":{
            "no":{}
        },
        "weak":{
            "yes":{}
        }
    })
})
tree.travel()


