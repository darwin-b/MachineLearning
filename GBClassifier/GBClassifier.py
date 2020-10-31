from sklearn import svm, metrics
from sklearn.datasets import fetch_openml, make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

print("-------------------Fetching data-------------------------------")
X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X/255
print("----------------Reading Data Complete-------------------------")

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# X, y = make_classification(random_state=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

scores=[]

max_features = ["auto",'sqrt','log2']
estimators = [10,15,20,50]

results ="\n"

for estimator in estimators:
    for max_feature in max_features:

        results += "---------------------------------------------------------------------------\n"
        results += "---------------------------------------------------------------------------\n"
        results += " No of Boosting Stages: " + str(estimator) + "\n"
        results += " split feature : " + str(max_feature) + "\n"

        print("\n---------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------")
        print(" No of Boosting Stages: ", estimator)
        print(" split feature : ", max_feature)

        clf = GradientBoostingClassifier(random_state=0, max_features=max_feature,n_estimators=estimator)
        clf.fit(X_train, y_train)

        clf.predict(X_test[:2])
        score = clf.score(X_test, y_test)
        scores.append(score)

        print("\n Test Accuracy  score: ", score)
        print("\n---------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------")

        results += "\nTest Accuracy  score:" + str(score)
        results += "\n---------------------------------------------------------------------------"
        results += "\n---------------------------------------------------------------------------\n\n"

with open("GBClassifier_resultsMatrix.txt",'w') as file:
    file.write(results)