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


mat = np.loadtxt("TP1_train.tsv",delimiter='\t')
data = shuffle(mat)
Ys = data[:,4].astype(int)
Xs = data[:,0:4]
means = np.mean(Xs,axis=0)
stdevs = np.std(Xs,axis=0)
Xs = (Xs-means)/stdevs


def bayes(X,Y, train_ix, valid_ix, bandwidth):
    t_0 = X[Y == 0,:]
    t_1 = X[Y == 1,:]
    
    for i in range(0,len(t_0),1):
        vals = t_0[i]
        if i == 0 :
            vals = np.log( abs(vals) )
        else :
            vals = np.log( abs(vals) )+t_0[i-1]  
        t_0[i] = vals
        
    for i in range(0,len(t_1),1):
        vals = t_1[i]
        if i == 0 :
            vals = np.log( abs(vals) )
        else :
            vals = np.log( abs(vals) ) + t_1[i-1]  
        t_1[i] = vals
    
    
    kde_0_0 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(t_0[:,0].reshape(-1, 1))    
    kde_0_1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(t_0[:,1].reshape(-1, 1))
    kde_0_2 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(t_0[:,2].reshape(-1, 1))
    kde_0_3 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(t_0[:,3].reshape(-1, 1))
    kde_1_0 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(t_1[:,0].reshape(-1, 1))
    kde_1_1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(t_1[:,1].reshape(-1, 1))
    kde_1_2 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(t_1[:,2].reshape(-1, 1))
    kde_1_3 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(t_1[:,3].reshape(-1, 1))
     
    print (np.argmax(kde_0_0.score_samples(t_0[:,0].reshape(-1, 1)))
            ,np.argmax(kde_0_1.score_samples(t_0[:,1].reshape(-1, 1)))
            ,np.argmax(kde_0_2.score_samples(t_0[:,2].reshape(-1, 1)))
            ,np.argmax(kde_0_3.score_samples(t_0[:,3].reshape(-1, 1)))
            ,np.argmax(kde_1_0.score_samples(t_0[:,0].reshape(-1, 1)))
            ,np.argmax(kde_1_1.score_samples(t_0[:,1].reshape(-1, 1)))
            ,np.argmax(kde_1_2.score_samples(t_0[:,2].reshape(-1, 1)))
            ,np.argmax(kde_1_3.score_samples(t_0[:,3].reshape(-1, 1)))
            )
    
X_r,X_t,Y_r,Y_t = train_test_split(Xs, Ys, test_size=0.33, stratify = Ys)
    
folds = 5
stratKf = StratifiedKFold( n_splits = folds)    
'''
for b in np.arange(0.01,1,0.02): 
    tr_err = va_err = 0 
    for tr_ix, val_ix in stratKf.split(Y_r, Y_r):
        r, v = bayes(X_r,  Y_r, tr_ix, val_ix,b)
        tr_err += r
        va_err += v    

'''   
bayes(X_r,  Y_r, 0, 0,1) 