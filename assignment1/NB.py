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

      
def calc_folds_bayes(X,Y, train_ix, val_ix, bandwidth):
    X_r = X[train_ix]
    Y_r = Y[train_ix]
    X_v = X[val_ix]
    Y_v = Y[val_ix]
    r,v = bayes(X_r,Y_r, X_v, Y_v, bandwidth)
    return r,v

def bayes(X_r,Y_r, X_v, Y_v, bandwidth):     
    kde = KernelDensity(bandwidth=bandwidth,kernel='gaussian')    
    
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
best_err = 1e12
best_bw = 1
bws = [round(b,3) for b in np.arange(0.02,0.6,0.02) ]
for bandwidth in bws: 
    tr_err = va_err = 0 
   
    for tr_ix, val_ix in stratKf.split(Y_r, Y_r):
        r,v = calc_folds_bayes(X_r,Y_r, tr_ix,val_ix, bandwidth) 
        tr_err += 1 - accuracy_score(r , Y_r[tr_ix])
        va_err += 1 - accuracy_score(v , Y_r[val_ix])
       
    tr_err = tr_err/folds
    va_err = va_err/folds
    errorTrain.append(tr_err)
    errorValidation.append(va_err)  

    if va_err < best_err:
        best_err = va_err
        best_bw = bandwidth


plt.figure(figsize=(8,8), frameon=True)
ax_lims=(-3,3,-3,3)
plt.axis(ax_lims)
plt.subplot(211)

plt.title("Naive Bayes with best Bandwidth: "+str(best_bw))

line1, = plt.plot(bws,errorTrain, label="Train Err", linestyle='-', color='blue')
line2, = plt.plot(bws,errorValidation, label="Validation Err", linestyle='-', color='green')

legend = plt.legend(handles=[line1,line2], loc='lower right')

ax = plt.gca().add_artist(legend)

plt.show()

plt.savefig('NB.png', dpi=300)

plt.close()
   
mat = np.loadtxt("TP1_test.tsv",delimiter='\t')
data = shuffle(mat)
Y_v = data[:,4].astype(int)
X_v = data[:,0:4]

r,v = bayes(Xs,Ys, X_v,Y_v, best_bw)
error = 1 - accuracy_score(v , Y_v)
print("Best Bandwidth Found "+str(best_bw)+" with Error of",error)
