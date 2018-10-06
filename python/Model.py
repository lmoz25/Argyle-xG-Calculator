import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from Functions import pca_transform, kFoldValidationScore, getData

X , t = getData()

for n in range(1,5):
    #dimensionality reduction
    X_trans, ratios = pca_transform(X, nComponents = n)

    #SVM
    clf = svm.SVC(probability=True)
    print("SVM accuracy with", n,"components: ", kFoldValidationScore(X_trans,t,clf,10))

    #k-NN
    kNN = KNeighborsClassifier(n_neighbors=12)
    print("kNN accuracy with", n, "components: ", kFoldValidationScore(X_trans,t,kNN,10))
    #decision tree
    tree = DecisionTreeClassifier()
    print("Tree accuracy with", n, "components: ", kFoldValidationScore(X_trans,t,tree,10))

    print(ratios)

