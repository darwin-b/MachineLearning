from sklearn import svm, metrics
from sklearn.datasets import  fetch_openml
from sklearn.neural_network import MLPClassifier

print("-------------------Fetching data-------------------------------")
X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X/255
print("----------------Reading Data Complete-------------------------")

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


# layer_sizes=[(10,),(25,),(50,),(75,),(100,),(250,)]
# max_iters = [5,10,25,50]

layer_sizes=[(10,),(25,)]
max_iters = [5,10,20]
solvers = ["sgd","adam","lbfgs"]



for layer_size in layer_sizes:
    for iter in max_iters:
        for solver in solvers:

            print("\n---------------------------------------------------------------------------")
            print("-----------------------------------------------------------------------------")
            print(" layer_size: ",layer_size)
            print(" Max_iterations: ",iter)
            print(" Solver: ", solver)

            mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=25, alpha=1e-4,
                                solver='sgd', verbose=10, random_state=1,
                                learning_rate_init=.1)
            mlp.fit(X_train, y_train)
            print("\nTrain set score: %f" % mlp.score(X_train, y_train))
            print("\nTest set score: %f" % mlp.score(X_train, y_train))
            print("\n---------------------------------------------------------------------------")
            print("-----------------------------------------------------------------------------")