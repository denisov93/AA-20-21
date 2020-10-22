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
from sklearn.metrics import accuracy_score


mat = np.loadtxt("TP1_train.tsv",delimiter='\t')
data = shuffle(mat)
Ys = data[:,4].astype(int)
Xs = data[:,0:4]
means = np.mean(Xs,axis=0)
stdevs = np.std(Xs,axis=0)
Xs = (Xs-means)/stdevs


def bayes(X,Y, train_ix, val_ix, bandwidth):
     
    kde = KernelDensity(bandwidth=bandwidth,kernel='gaussian')
    
    #fit   
    X_r = X[train_ix]
    Y_r = Y[train_ix]
    X_v = X[val_ix]
    Y_v = Y[val_ix]
    
    t_0 = X_r[Y_r == 0,:] #real
    t_1 = X_r[Y_r == 1,:] #fakes
    v_0 = X_v[Y_v == 0, :] 
    v_1 = X_v[Y_v == 1, :] 
    
    
    # log(A/ (A + B ) )
    p_0 =  np.log( t_0.shape[0] / X_r.shape[0] )
    p_1 =  np.log( t_1.shape[0] / X_r.shape[0] )       
    pv_0 = np.log( v_0.shape[0] / X_v.shape[0] )
    pv_1 = np.log( v_1.shape[0] / X_v.shape[0] )
    
    
    sum_logs_t_0 = np.ones(X_r.shape[0]) * p_0
    sum_logs_t_1 = np.ones(X_r.shape[0]) * p_1   
    sum_logs_v_0 = np.ones(X_v.shape[0]) * pv_0
    sum_logs_v_1 = np.ones(X_v.shape[0]) * pv_1    
    
    classes = np.zeros(X_r.shape[0])
    classes_n = np.zeros(X_v.shape[0])
    
    for i in range(X_r.shape[1]):
        kde.fit(t_0[:,[i]])
        sum_logs_t_0 += kde.score_samples(X_r[:,[i]])
        sum_logs_v_0 += kde.score_samples(X_v[:,[i]])        

        kde.fit(t_1[:,[i]])
        sum_logs_t_1 += kde.score_samples(X_r[:,[i]])
        sum_logs_v_1 += kde.score_samples(X_v[:,[i]])
    
    
    classes[(sum_logs_t_1 > sum_logs_t_0)] = 1
     
    classes_n[(sum_logs_v_1 > sum_logs_v_0 )] = 1
    
    
            
    return classes,classes_n
    

   
X_r,X_t,Y_r,Y_t = train_test_split(Xs, Ys,test_size=0.33, stratify = Ys)

folds = 5
stratKf = StratifiedKFold( n_splits = folds)    
errorTrain = []
errorValidation = []
best_err = 0.02
best_bw = 1

for bandwidth in np.arange(0.02,0.6,0.02): 
    tr_err = va_err = 0 
   
    for tr_ix, val_ix in stratKf.split(Y_r, Y_r):
        r,v = bayes(X_r,Y_r, tr_ix,val_ix, bandwidth) 
        tr_err += 1 - accuracy_score(r , Y_r[tr_ix])
        va_err += 1 - accuracy_score(v , Y_r[val_ix])
       
    tr_err = tr_err/folds
    va_err = va_err/folds
    errorTrain.append(tr_err)
    errorValidation.append(va_err)  

    if va_err < best_err:
        best_err = tr_err
        best_bw = bandwidth


line1, = plt.plot(errorTrain, label="errorTrain", linestyle='-')
line2, = plt.plot(errorValidation, label="errorValidation", linestyle='-',color="green")

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
plt.show() 
plt.savefig('NB.png', dpi=300)
plt.close() 
   
