import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from os import listdir

def pca_transform(data, nComponents=None):
    pca = PCA(n_components=nComponents)
    return pca.fit_transform(data), pca.fit(data).explained_variance_ratio_

def kFoldValidationScore(X, t, model, nFolds):
    kf = KFold(n_splits=nFolds)
    total = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        t_train, t_test = t.iloc[train_index], t.iloc[test_index]
        model.fit(X_train,t_train)
        total += model.score(X_test,t_test)
    return total/10

def getData():

    ''' 
    Function to extract shot data from CSV files stored in /Data to train models with.\n
    Input: None\n
    Returns: Tuple (X,t)

    '''

    keys = ['Zone', 'Foot','Assist', 'Marking', 'Result']
    shotData = pd.DataFrame(data=dict.fromkeys(keys,None), index=[0])

    for fl in listdir("../Data"):
        print(fl)
        if fl.endswith('.csv'):
            shotData = shotData.append(pd.read_csv("../Data/" + fl, usecols=keys), ignore_index=True)
    
    shotData = shotData.drop(shotData.index[0])
    
    #turn data into something the SVM can actually use
    shotData['Foot'].replace(['S','W','H','O'],[0,1,2,3],inplace = True)
    shotData['Assist'].replace(['O','S','F','C','P','D','R'],[0,1,2,3,4,5,6],inplace = True)
    shotData['Result'].replace(['G','W','S','O','B'],[0,1,2,3,4],inplace = True)

    return shotData[['Zone', 'Foot','Assist', 'Marking']], shotData['Result']