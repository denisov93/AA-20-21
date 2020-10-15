# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:47:14 2020

@author: adeni
"""

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

def calc_fold(X,Y, train_ix,valid_ix,C):
    """return error for train and validation sets
        
    """

    reg = LogisticRegression(C=C, tol=1e-10)
    reg.fit(X[train_ix],Y[train_ix])
    erroVal = 1 - reg.score(X[valid_ix],Y[valid_ix])
    erroTreino =  1 - reg.score(X[train_ix],Y[train_ix])
    return (erroTreino,erroVal)

mat = np.loadtxt("TP1_train.tsv",delimiter='\t')
data = shuffle(mat)
Ys = data[:,4].astype(int)
Xs = data[:,0:4]
means = np.mean(Xs,axis=0)
stdevs = np.std(Xs,axis=0)
Xs = (Xs-means)/stdevs

mat = np.loadtxt("TP1_test.tsv",delimiter='\t')
data = shuffle(mat)
Y_t = data[:,4].astype(int)
X_t = data[:,0:4]
means = np.mean(X_t,axis=0)
stdevs = np.std(X_t,axis=0)
X_t = (X_t-means)/stdevs


#X_r,X_t,Y_r,Y_t = train_test_split(Xs, Ys, test_size=0.33, stratify = Ys)

folds = 5
stratKf = StratifiedKFold( n_splits = folds)

errorTrain = []
errorValidation = []
c = 1e5

for feats in range(2,6):
    tr_err = va_err = 0
    for tr_ix, val_ix in stratKf.split(Ys, Ys):
        r, v = calc_fold(feats, Xs,  Ys, tr_ix, val_ix,c)
        tr_err += r
        va_err += v
    errorTrain.append(tr_err/folds)
    errorValidation.append(va_err/folds)
    print(feats, ':', tr_err/folds, va_err/folds)
print("accuracy:",(sum(errorTrain)+sum(errorValidation))/3)        
    
line1, = plt.plot(errorTrain, label="errorTrain", linestyle='--')
line2, = plt.plot(errorValidation, label="errorValidation", linestyle='--',color="red")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
#plt.show() 
    
#plt.savefig('final_plot.png', dpi=300)
#plt.close()