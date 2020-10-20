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

def make_hists(data,features):
    hists = []
    for feat in features:
        hists.append(np.ones(len(feat)))
    for ix in range(len(hists)):
        hists[ix] = np.log(hists[ix]/float(data.shape[0]+len(features[ix])))
    return hists
    
def split_data(features,test_fraction): 
     oks = Xs[Ys==0,:]
     noks = Xs[Ys==1,:]
     e_test_points = int(test_fraction*oks.shape[0])
     e_train = oks[e_test_points:,:]
     e_test = oks[:e_test_points,:]
     p_test_points = int(test_fraction*noks.shape[0])
     p_train = noks[p_test_points:,:]
     p_test = noks[:p_test_points,:]
     return e_train,p_train,e_test,p_test

def classify(e_class,e_log,p_class,p_log,feat_mat):
     classes = np.zeros(feat_mat.shape[0])
     for row in range(feat_mat.shape[0]):
         e_sum = e_log
         p_sum = p_log
         for column in range(feat_mat.shape[1]):
             e_sum = e_sum + e_class[column][int(feat_mat[row,column])]
             p_sum = p_sum + p_class[column][int(feat_mat[row,column])]
         if e_sum<p_sum:
             classes[row]=1
     return classes

def do_bayes():
     features = Xs
     
     e_train,p_train,e_test,p_test = split_data(features,0.33)
     e_hists = make_hists(e_train,features)
     
     p_hists = make_hists(p_train,features)
     tot_len = e_train.shape[0]+p_train.shape[0]
     e_log = np.log(float(e_train.shape[0])/tot_len)
     p_log = np.log(float(p_train.shape[0])/tot_len)
     c_e = classify(e_hists,e_log,p_hists,p_log,e_test)
     c_p = classify(e_hists,e_log,p_hists,p_log,p_test)
     errors = sum(c_e)+sum(1-c_p)
     error_perc = float(errors)/(len(c_e)+len(c_p))*100
     print(f'{errors:.0f} errors; {error_perc:.2f}% error rate')
     print('\tE\tP')
     print(f'E\t{sum(1-c_e):.0f}\t{sum(1-c_p):.0f}')
     print(f'P\t{sum(c_e):.0f}\t{sum(c_p):.0f}')

    
from sklearn.naive_bayes import GaussianNB


mata = np.loadtxt("TP1_test.tsv",delimiter='\t')
dataa = shuffle(mata)
Y_t = dataa[:,4].astype(int)
X_t = dataa[:,0:4]
X_t = (X_t-means)/stdevs

clf = GaussianNB()
clf.fit(Xs, Ys)
print(clf.score(X_t,Y_t))



do_bayes()