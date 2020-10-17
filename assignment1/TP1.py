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
3) Curtosis
4) Entropy
5) Class Label [0=RealBankNotes & 1=FakeBankNotes]

Classifiers(3) needed for the project:
> Logistic Regression
> Naïve Bayes Classifier using Kernel Density Estimation
> Gaussian Naïve Bayes classifier

Comparing classifiers: 
> Normal test (95% confidence)
> McNemar's test (95% confidence)

Observations:
> "Não esqueçam que, no caso do Naive Bayes, kde.fit deve ser feito para cada "par" (Classe, Atributo)."
'''

##Region Imports
#
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from TP1_aux import poly_mat
from TP1_aux import create_plot
#
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity #reminder: Needs to find the optimum value for the bandwitdh parameter of the kernel density estimators
from sklearn.naive_bayes import GaussianNB #reminder: no parameter adjustment
#
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
##End of Region Imports

#Debug
time_ms = lambda: int(round(time.time() * 1000))
start = time_ms()

def sep(text):
    print("~~~~~~~ "+text+" ~~~~~~~~~~~~~~~~~")
def sepn(text):
    sep(text)
    print("\n")

def showLoadShuffleDebug():
    sep("Tests [V S C E Class]")
    print(tests)
    print("Total: "+str(len(tests)))
    sep("Train [V S C E Class]")
    print(train)
    print("Total: "+str(len(train)))
    sepn("Loading & Shuffle: Complete")

#

#Other
'''
def calc_fold(feats, X,Y, train_ix,valid_ix,C=1e12):
    """return error for train and validation sets"""
    reg = LogisticRegression(C=C, tol=1e-10)
    reg.fit(X[train_ix,:feats],Y[train_ix])
    prob = reg.predict_proba(X[:,:feats])[:,1]
    squares = (prob-Y)**2
    return np.mean(squares[train_ix]),np.mean(squares[valid_ix])'''

def calc_fold(X,Y, train_ix,valid_ix,C):
    """return error for train and validation sets    """
    reg = LogisticRegression(C=C, tol=1e-10)
    reg.fit(X[train_ix],Y[train_ix])
    erroVal = 1 - reg.score(X[valid_ix],Y[valid_ix])
    erroTreino =  1 - reg.score(X[train_ix],Y[train_ix])
    return (erroTreino, erroVal)

#File Loading
def load_file(file):
    matrix = np.loadtxt(file,delimiter='\t')
    return matrix
tests = load_file("TP1_test.tsv")
train = load_file("TP1_train.tsv")

#Shuffle
tests = shuffle(tests)
train = shuffle(train)
showLoadShuffleDebug()

#Standardizing
sep("Standardizing")
#Train
Ys = train[:,4].astype(int)
Xs = train[:,0:4]
means = np.mean(Xs,axis=0)
stdevs = np.std(Xs,axis=0)
Xs = (Xs-means)/stdevs
#Tests
Y_finaltest = tests[:,4].astype(int)
X_finaltest = tests[:,0:4]
finaltest_means = np.mean(X_finaltest,axis=0)
finaltest_stdevs = np.std(X_finaltest,axis=0)
X_finaltest = (X_finaltest-finaltest_means)/finaltest_stdevs

print("Preparing training set")
print(Ys)
print(Xs)
print("Preparing set for final test")
print(Y_finaltest)
print(X_finaltest)
sepn("Standardizing: Complete")


sep("StratifiedKFold")

X_r,X_t,Y_r,Y_t = train_test_split(Xs, Ys, test_size=0.33, stratify = Ys)

folds = 5
stratKf = StratifiedKFold( n_splits = folds)

errorTrain = []
errorValidation = []
best_feats = -1
best_re = 1e12
C = 5

for feats in range(1,6):
    tr_err = va_err = 0
    for tr_ix, val_ix in stratKf.split(Y_r, Y_r):
        r, v = calc_fold(X_r, Y_r, tr_ix, val_ix, C)
        tr_err += r
        va_err += v
    errorTrain.append(tr_err/folds)
    errorValidation.append(va_err/folds)
    print(feats, ':', tr_err/folds, va_err/folds)
    re = va_err/folds - tr_err/folds
    if(re < best_re):
        best_feats = feats
        best_re = re
        print("New best feature: "+str(best_feats))

print("Best C: "+str(best_re))
sep("End of best features ploting")

plt.figure(figsize=(8,8), frameon=True)
ax_lims=(-3,3,-3,3)
plt.axis(ax_lims)
plt.subplot(211)

line1, = plt.plot(errorTrain, label="Train Err", linestyle='--', color='blue')
line2, = plt.plot(errorValidation, label="Validation Err", linestyle='--', color='green')

legend = plt.legend(handles=[line1,line2], loc='upper right')

ax = plt.gca().add_artist(legend)
plt.savefig('error_validation_plot.png', dpi=300)
plt.show()
plt.close()

#Process Finish
end = time_ms()
runtime = end - start
print("Runtime: "+str(runtime)+"ms")





