
import sys
import numpy as np
import math
from scipy.sparse.csgraph import minimum_spanning_tree
import random
import timeit
# import networkx as nx
from scipy.sparse import csr_matrix


# computing immediate parent on DFS
def dfs(graph, start):
    parents = np.zeros((1,len(graph)),dtype=int)
    visited, stack = set(), [start],
    parents[0,start]=-1
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            new_vertexes = graph[vertex] - visited
            stack.extend(new_vertexes)
            for i in new_vertexes:
                parents[0,i]=vertex
    return parents


#
# trainingFileName = sys.argv[1]
# testingFileName= sys.argv[2]

root_path = sys.argv[1]+"\\"


# root_path = "C:\\Users\\darwi\\OneDrive - The University of Texas at Dallas\\Acads\\Machine Learning\\Assignments\\MachineLearning\\Data\\small-10-datasets\\"
# train_file = "dna.ts"
# test_file = "dna.test"
# extension = ".data"

# datasets=["accidents","baudio","bnetflix","dna","jester","kdd","msnbc","nltcs","plants"]

datasets=["plants","nltcs","msnbc","kdd","jester","dna","bnetflix","baudio","accidents"]

# datasets=["dna"]

for dataset in datasets:


    print("------------------------- ",dataset," -------------------------------")

    x_train = np.loadtxt(root_path + dataset + ".ts.data", delimiter=',', dtype=int)
    x_test = np.loadtxt(root_path + dataset + ".test.data", delimiter=',', dtype=int)

    start = timeit.default_timer()

    n_test = x_test.shape[0]
    n_size = x_train.shape[0]
    n_features = x_train.shape[1]
    print("Size : ", n_size, " X ", n_features)
    p_array = np.zeros((2, n_features))
    p_array[1] = (x_train.sum(axis=0) + 1) / (n_size + 2)
    p_array[0] = 1 - p_array[1]



    # -------------------weights----------------------------


    I = np.zeros((n_features,n_features))
    sum_ds = {}

    for j in range(0,n_features):
        sum_ds[j]={}
        for k in range(j+1,n_features):
            c = np.ones((2,2))
            # c= c/(n_size+4)
            for i in range(0,n_size):
                c[x_train[i,j],x_train[i,k]] += 1

            c /= (n_size + 4)

            x = (p_array[1,j])*(p_array[0,k])
            I[j][k] += c[1,0]*np.log2((c[1,0]/x))

            x = (p_array[1,j])*(p_array[1,k])
            I[j][k] += c[1,1]*np.log2((c[1,1]/x))

            x = (p_array[0,j])*(p_array[0,k])
            I[j][k] += c[0,0]*np.log2((c[0,0]/x))

            x = (p_array[0,j])*(p_array[1,k])
            I[j][k] += c[0,1]*np.log2((c[0,1]/x))

            I[j][k] = -1*I[j][k]

            c *= (n_size + 4)
            sum_ds[j][k]=[[c[0,0],c[0,1]],[c[1,0],c[1,1]]]


    m_tree = minimum_spanning_tree(csr_matrix(I)).toarray()

    # g1 = nx.DiGraph(m_tree)

    # spanning tree to graph
    nodes = [node for node in range(0, n_features)]
    G = {key: set() for key in nodes}

    for i in range(0, n_features):
        col = m_tree[i].nonzero()[0]
        l=len(col)
        if (0 < l):
            for j in range(0,l):
                # add edges
                G[i].add(col[j])
                G[col[j]].add(i)


    # randomize starting dfs node
    Random_startNode = random.randint(0,n_features-1)
    parents = dfs(G,Random_startNode )

    # --------------Predict------------------------------

    test_instance = 0
    c=0
    for i in range(0, n_test):
        for j in range(0, n_features):
            p = parents[0, j]
            if p==-1:
                test_instance += np.log2(p_array[x_test[i, j], j])
            try:
                c = sum_ds[p][j][x_test[i, p]][x_test[i, j]]
            except:
                try:
                    c = sum_ds[j][p][x_test[i, j]][x_test[i, p]]
                except:
                    temp="ignore"
            joint = c /(n_size + 4)
            test_instance += np.log2(joint / p_array[x_test[i, p], p])


    log_likelihood = test_instance /n_test
    print('Log Likelihhod: ', log_likelihood)

    # -------------------------------

    end = timeit.default_timer()
    print("Execution Time: ",end - start)
    print("")
    print("-----------------------------------------------------------------")
