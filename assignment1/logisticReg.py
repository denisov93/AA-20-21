# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:47:14 2020

@author: adeni
"""

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

def poly_mat(reg,X_data,feats,ax_lims):
    """create score matrix for contour
    """
    Z = np.zeros((200,200))
    xs = np.linspace(ax_lims[0],ax_lims[1],200)
    ys = np.linspace(ax_lims[2],ax_lims[3],200)
    X,Y = np.meshgrid(xs,ys)
    points = np.zeros((200,2))
    points[:,0] = xs
    for ix in range(len(ys)):
        points[:,1] = ys[ix]
        x_points=poly_16features(points)[:,:feats]
        Z[ix,:] = reg.decision_function(x_points)
    return (X,Y,Z)

def calc_fold(feats, X,Y, train_ix,valid_ix,C=1e12):
    """return error for train and validation sets"""
    reg = LogisticRegression(C=C, tol=1e-10)
    reg.fit(X[train_ix,:feats],Y[train_ix])
    prob = reg.predict_proba(X[:,:feats])[:,1]
    squares = (prob-Y)**2
    return np.mean(squares[train_ix]),np.mean(squares[valid_ix])

def create_plot(plt,X_r, Y_r, X_t, Y_t, feats, best_c):
    """create imege with plot for best classifier"""
  #  ax_lims=(-3,3,-3,3)
   # plt.figure(figsize=(8,8), frameon=False)
   # plt.axis(ax_lims)
    reg = LogisticRegression(C=best_c, tol=1e-10)
    reg.fit(X_r[:,:feats],Y_r)
    plotX,plotY,Z = poly_mat(reg,X_r,feats,ax_lims)
    plt.contourf(plotX,plotY,Z,[-1e16,0,1e16], colors = ('b', 'r'),alpha=0.05)
    plt.contour(plotX,plotY,Z,[0], colors = ('k'))
    plt.plot(X_r[Y_r>0,0],X_r[Y_r>0,1],'or')
    plt.plot(X_r[Y_r<=0,0],X_r[Y_r<=0,1],'ob')
    plt.plot(X_t[Y_t>0,0],X_t[Y_t>0,1],'xr',mew=2)
    plt.plot(X_t[Y_t<=0,0],X_t[Y_t<=0,1],'xb',mew=2)
 #   plt.savefig('final_plot.png', dpi=300)
 #   plt.close()

def poly_16features(X):
    """Expand data polynomially
    """
    X_exp = np.zeros((X.shape[0],X.shape[1]+14))
    X_exp[:,:2] = X 
    X_exp[:,2] = X[:,0]*X[:,1]
    X_exp[:,3] = X[:,0]**2
    X_exp[:,4] = X[:,1]**2
    X_exp[:,5] = X[:,0]**3
    X_exp[:,6] = X[:,1]**3
    X_exp[:,7] = X[:,0]**2*X[:,1]
    X_exp[:,8] = X[:,1]**2*X[:,0]
    X_exp[:,9] = X[:,0]**4
    X_exp[:,10] = X[:,1]**4
    X_exp[:,11] = X[:,0]**3*X[:,1]
    X_exp[:,12] = X[:,1]**3*X[:,0]
    X_exp[:,13] = X[:,0]**2*X[:,1]**2
    X_exp[:,14] = X[:,0]**5
    X_exp[:,15] = X[:,1]**5        
    return X_exp


#standartization  "TP1_train.tsv"
#def standart(file):
mat = np.loadtxt("TP1_train.tsv",delimiter='\t')
data = shuffle(mat)
Ys = data[:,4].astype(int)
Xs = data[:,0:4]
means = np.mean(Xs,axis=0)
stdevs = np.std(Xs,axis=0)
Xs = (Xs-means)/stdevs
#return (Xs,Ys)
mat = np.loadtxt("TP1_test.tsv",delimiter='\t')
data = shuffle(mat)
Y_t = data[:,4].astype(int)
X_t = data[:,0:4]
means = np.mean(X_t,axis=0)
stdevs = np.std(X_t,axis=0)
X_t = (X_t-means)/stdevs

#expand
#Xs = poly_16features(Xs)
#X_t = poly_16features(X_t)
#X_r,X_t,Y_r,Y_t = train_test_split(Xs, Ys, test_size=0.33, stratify = Ys)
trans = PolynomialFeatures(4, interaction_only=True, include_bias=False)
Xs = trans.fit_transform(Xs)
X_t = trans.fit_transform(X_t)

folds = 10
stratKf = StratifiedKFold( n_splits = folds)

errorTrain = []
errorValidation = []
best_feats = 2
best_re = 1e12 

ax_lims=(-3,3,-3,3)
plt.figure(figsize=(8,8), frameon=False)
plt.axis(ax_lims)

for feats in range(2,16):
    tr_err = va_err = 0
    for tr_ix, val_ix in stratKf.split(Ys, Ys):
        r, v = calc_fold(feats, Xs,  Ys, tr_ix, val_ix)
        tr_err += r
        va_err += v
    errorTrain.append(tr_err/folds)
    errorValidation.append(va_err/folds)
    print(feats, ':', tr_err/folds, va_err/folds)
    re = va_err/folds - tr_err/folds
    if(re < best_re):
        best_feats = feats
        best_re = re        
    create_plot(plt,Xs, Ys, X_t, Y_t, feats, re)

   
    
plt.savefig('final_plot.png', dpi=300)
plt.close()