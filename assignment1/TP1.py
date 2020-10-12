'''
Assignment 1 by
Alexander Denisov (44592)
Samuel Robalo (41936)
AA 20/21
TP4 Instructor: Joaquim Francisco Ferreira da Silva
Regency: Ludwig Krippahl
'''

'''
TP1 Test & Train File contents
Features(4) + ClassLabels(1):
1) Variance 
2) Skewness 
3) Curtosis of Wavelet Transformed image
4) Entropy of the bank note image
5) Class Label [0=RealBankNotes & 1=FakeBankNotes]

Classifiers(3) needed for the project:
> Logistic Regression
> Naïve Bayes Classifier using Kernel Density Estimation
> Gaussian Naïve Bayes classifier

Comparing classifiers: 
> Normal test (95% confidence)
> McNemar's test (95% confidence)

Observations:
>

'''

##Region Imports
#
import numpy as np
import matplotlib.pyplot as plt
#
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity #reminder: Needs to find the optimum value for the bandwitdh parameter of the kernel density estimators
from sklearn.naive_bayes import GaussianNB #reminder: no parameter adjustment
#
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
##End of Region Imports

#File Loading
def load_file(file):
    matrix = np.loadtxt(file,delimiter='\t')
    return matrix

def calc_fold(feats, X,Y, train_ix,valid_ix,C=1e12):
    """return error for train and validation sets"""
    reg = LogisticRegression(C=C, tol=1e-10)
    reg.fit(X[train_ix,:feats],Y[train_ix])
    prob = reg.predict_proba(X[:,:feats])[:,1]
    squares = (prob-Y)**2
    return np.mean(squares[train_ix]),np.mean(squares[valid_ix])


#File Loading
def load_file(file):
    matrix = np.loadtxt(file,delimiter='\t')
    return matrix
tests = load_file("TP1_test.tsv")
train = load_file("TP1_train.tsv")

#Shuffle
np.random.shuffle(tests)
np.random.shuffle(train)

print("tests [V S C E Class]")
print(tests)
print("train [V S C E Class]")
print(train)










