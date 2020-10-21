# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:47:14 2020

@author: adeni
"""
'''
com paramentros de C obtemos valores com erro de perto de 10% seria possivel diminuir?
'''
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.linear_model import LogisticRegression

def calc_fold(X,Y, train_ix,valid_ix,C):    
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


best_param_C = []

c_par = [1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e7,1e8,1e9,1e10,1e11,1e12]

X_r,X_t,Y_r,Y_t = train_test_split(Xs, Ys, test_size=0.33, stratify = Ys)

folds = 5
stratKf = StratifiedKFold( n_splits = folds)
errorTrain = []
errorValidation = []
counter = 0
ind = 0
smaller = 1
cs = []

for c in c_par: 
    tr_err = va_err = 0 
    for tr_ix, val_ix in stratKf.split(Y_r, Y_r):
        r, v = calc_fold(X_r,  Y_r, tr_ix, val_ix,c)
        tr_err += r
        va_err += v
    
    cs.append(c)
    if(smaller>va_err/folds):
        smaller = va_err/folds
        ind = counter
        
    counter+=1
    errorTrain.append(tr_err/folds)
    errorValidation.append(va_err/folds)
    print(c, ':', tr_err/folds, va_err/folds)

      
print("media C's : ",sum(errorValidation)/counter)
print("escolhido :", cs[ind])    
line1, = plt.plot(errorTrain, label="errorTrain", linestyle='--')
line2, = plt.plot(errorValidation, label="errorValidation", linestyle='--',color="red")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
plt.show() 

mat = np.loadtxt("TP1_test.tsv",delimiter='\t')
data = shuffle(mat)
Y_t = data[:,4].astype(int)
X_t = data[:,0:4]

X_t = (X_t-means)/stdevs

reg = LogisticRegression(C=cs[ind], tol=1e-10)
reg.fit(Xs, Ys)
erroVal = 1 - reg.score(X_t,Y_t)
print("resultado do teste erro de avaliação:",erroVal)

    
#plt.savefig('final_plot.png', dpi=300)
#plt.close()