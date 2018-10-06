import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

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
    Function to extract shot data from Shots.csv and xG database - Sheet1.csv.\n
    Input: flag whether result field should distinguish between different types of misses (mixed) or not (simple)\n
    Returns: Tuple (X,t)

    '''
    #bring in shot data from both sources and combine
    shotData = pd.read_csv("Shots.csv", usecols=['Zone', 'Foot', 'Assist', 'Marking', 'Result'])
    shotData = shotData.append(pd.read_csv("xG_database_2018_19 - Data.csv", usecols=['Zone', 'Foot', 'Assist', 'Marking', 'Result']), ignore_index=True)
    shotData = shotData.append(pd.read_csv("xG_database_2017_18 - Data.csv", usecols=['Zone', 'Foot', 'Assist', 'Marking', 'Result']), ignore_index=True)
    #turn data into something the SVM can actually use
    shotData['Foot'].replace(['S','W','H','O'],[0,1,2,3],inplace = True)
    shotData['Assist'].replace(['O','S','F','C','P','D','R'],[0,1,2,3,4,5,6],inplace = True)
    shotData['Result'].replace(['G','W','S','O','B'],[0,1,2,3,4],inplace = True)

    return shotData[['Zone', 'Foot','Assist', 'Marking']], shotData['Result']
