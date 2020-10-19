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

def printProgressBar (iteration, total, decimals = 1, printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    print(f'\r({percent}%)', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
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
finaltest_means = means
finaltest_stdevs = stdevs
X_finaltest = (X_finaltest-finaltest_means)/finaltest_stdevs

print("Preparing training set test/validation")
X_train,X_test,Y_train,Y_test = train_test_split(Xs, Ys, test_size=0.33, stratify = Ys)
print(X_train)
print(Y_train)
print("Preparing set for final test")
print(Y_finaltest)
print(X_finaltest)
sepn("Standardizing: Complete")

sep("StratifiedKFold")

folds = 5
stratKf = StratifiedKFold( n_splits = folds)
errorTrain = []
errorValidation = []

best_re = 1e12
best_C = 1e12

c_from = 0.0001
c_to = 0.01
step = 0.000025
calc = int((c_to - c_from)/step)
counter = 0
print("Calculating "+str(calc)+" values for C")

for c in np.arange(c_from,c_to,step):
    counter += 1
    tr_err = va_err = 0
    printProgressBar(counter,calc)
    for tr_ix, val_ix in stratKf.split(Y_train, Y_train):
        r, v = calc_fold(X_train, Y_train, tr_ix, val_ix, c)
        tr_err += r
        va_err += v
    errorTrain.append(tr_err/folds)
    errorValidation.append(va_err/folds)
    
    if(va_err > tr_err):
        re = (va_err - tr_err)
    else: 
        re = (tr_err - va_err)
    
    if(re < best_re):
        best_re = re
        best_C = round(c,7)
        
print("Best C: "+str(best_C))
sep("End of best C ploting")

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

reg = LogisticRegression(C=1e10, tol=1e-10)
reg.fit(X_train, Y_train)
erroVal = 1 - reg.score(X_finaltest,Y_finaltest)
print(erroVal)

#Process Finish
end = time_ms()
runtime = end - start
print("Runtime: "+str(runtime)+"ms")