from sklearn import svm, metrics
from sklearn.datasets import  fetch_openml

print("-------------------Fetching data-------------------------------")
X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X/255
print("----------------Reading Data Complete-------------------------")

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]



c = [0.5,1,1.5,2]  # regularisation l2 parameter
degree = [2,3,4]   # if polynomial kernel
gamma =["scale","auo"] # if rbf Kernel
max_iter = [100,500]
# max_iter = [100,1000,5000,10000]

# clasifier = svm.SVC(gamma=0.001)
#
# clasifier.fit(X_train,y_train)
# pred = clasifier.predict(X_test)
# print(metrics.classification_report(y_test,pred))


'''
Linear Kernel
'''
for l2 in c:
    for iter in max_iter:
        classifier = svm.SVC(kernel="poly",C=l2,max_iter=iter)

        classifier.fit(X_train,y_train)
        pred = classifier.predict(X_test)
        print("\n------------   Linear Kernel   l2 Regulairstion: ",l2,"   max_iterations: ",iter,"--------------")
        print(metrics.classification_report(y_test,pred))
        print("-------------------------------------------------------------------------------------------------\n")


'''
polynomial Kernel
'''
for l2 in c:
    for iter in max_iter:
        classifier = svm.SVC(kernel="linear",C=l2,max_iter=iter)

        classifier.fit(X_train,y_train)
        pred = classifier.predict(X_test)
        print("\n------------   Polynomial Kernel (degree=3)   l2 Regulairstion: ",l2,"   max_iterations: ",iter,"--------------")
        print(metrics.classification_report(y_test,pred))
        print("-------------------------------------------------------------------------------------------------\n")


'''
RBF Kernel
'''
for l2 in c:
    for iter in max_iter:
        for g in gamma:
            classifier = svm.SVC(kernel="rbf",C=l2,max_iter=iter,gamma=g)

            classifier.fit(X_train,y_train)
            pred = classifier.predict(X_test)
            print("\n------------   RBF Kernel gamma:",g,"  l2 Regulairstion: ",l2,"   max_iterations: ",iter,"--------------")
            print(metrics.classification_report(y_test,pred))
            print("-------------------------------------------------------------------------------------------------\n")
