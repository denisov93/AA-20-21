# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 19:36:15 2020

@author: adeni
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from scipy.stats import norm

n_features = 4

mat = np.loadtxt("TP1_train.tsv",delimiter='\t')
data = shuffle(mat)
Ys = data[:,4].astype(int)
Xs = data[:,0:4]
means = np.mean(Xs,axis=0)
stdevs = np.std(Xs,axis=0)
Xs = (Xs-means)/stdevs


def bayes(X,Y, train_ix, valid_ix, bandwidth):
    
    #fit
    
    t_0 = X[Y == 0,:] #real
    t_1 = X[Y == 1,:] #fakes
    
    # log(A/ (A + B ) )
    p_a = t_0.shape[0]
    p_b = t_1.shape[1]
    p_uni = p_a + p_b
    
    p_0 =  np.log( p_a / p_uni )
    p_1 =  np.log( p_b / p_uni )
    
    features_0 = [] #features of real notes
    features_1 = [] #features of fake notes
    
    for i in range(n_features):
        features_0[i] = KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(t_0[:,i].reshape(-1,1))
        features_1[i] = KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(t_1[:,i].reshape(-1,1))
    
    #prediction
    

    
X_r,X_t,Y_r,Y_t = train_test_split(Xs, Ys, test_size=0.33, stratify = Ys)
    
folds = 5
stratKf = StratifiedKFold( n_splits = folds)    
errorTrain = []
errorValidation = []
best_err = 1e12
best_bw = 1

for bandwidth in np.arange(0.02,0.06,0.02): 
    tr_err = va_err = 0 
    for tr_ix, val_ix in stratKf.split(Y_r, Y_r):
        r, v = bayes(X_r,Y_r, tr_ix, val_ix,bandwidth)
        tr_err += r
        va_err += v    
    
    if tr_err < best_err:
        best_err = tr_err
        best_bw = bandwidth
        
    errorTrain.apend(tr_err)
    errorValidation.append(va_err)
   






























