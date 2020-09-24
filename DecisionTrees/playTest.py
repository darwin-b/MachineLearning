import numpy as np
import pandas as pd
import time



data = pd.read_csv("./Data/train_c300_d100.csv", header=None)

feature = 2
target = 500
t1 = data.groupby([feature,target])
t2 = t1.size()



data2 = pd.DataFrame({"Outlook":["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
                     "Temp":["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
                     "Humidity":["High","High","High","High","Normal","Normal","Normal","High","Normal","Normal","Normal","High","Normal","High"],
                     "Wind":["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
                     "Play":["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"]},
                    columns=["Outlook","Temp","Humidity","Wind","Play"])


feature2 = "Outlook"
target2 = "Play"
p1 = data2.groupby([feature2,target2])
p2 = p1.size()

print(t2)
print(p2)

for each in p1.indices:
    for x in each:
        print(x)
    print("-----------------")
    print(p1.indices[each])

print(p2.index)


feat={}
for each in p1.indices.keys():
    print(p1.indices[each])
    if each[0] in feat.keys():
        feat[each[0]][each[1]]=p1.indices[each]
    else:
        feat[each[0]]={each[1]:p1.indices[each]}
print(feat)

feat_count={}
for each in p2.keys():
    if each[0] in feat_count.keys():
        feat_count[each[0]][each[1]]=p2[each]
    else:
        feat_count[each[0]]={each[1]:p2[each]}
print(feat_count)


commonClass= data2.loc[:, target2].value_counts().idxmax()
commonClass= data2.loc[:, target2].value_counts()


print(data[target].value_counts())

child_label="Yes"
label = data2[data2.columns[-1]].value_counts()

print(label[child_label])