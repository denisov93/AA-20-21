# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:42:04 2020

@author: adeni
"""
import numpy as np
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB 


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

X_t = (X_t-means)/stdevs

gaus = GaussianNB()
gaus.fit(Xs, Ys)
erroVal = 1 - gaus.score(X_t,Y_t)
print("resultado do teste erro de avaliação:",erroVal)
