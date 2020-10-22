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
import numpy as np
import matplotlib.pyplot as plt
#
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity #reminder: Needs to find the optimum value for the bandwitdh parameter of the kernel density estimators
from sklearn.naive_bayes import GaussianNB #reminder: no parameter adjustment
#
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold

def sep(text):
    print("~~~~~~~ "+text+" ~~~~~~~~~~~~~~~~~")
def sepn(text):
    sep(text)
    print("\n")

def printProgressBar (iteration, total, decimals = 1):
    """Makes a % loading
    
    @params:
        iteration - Required : current iteration (Int)
        total     - Required : total iterations (Int)
        decimals  - Optional : number of decimals (Int)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    print(f'\r({percent}%)', end = "\r")
    if iteration == total: print()
#Logistic Regression Calc Folds
def calc_fold_logistic(X,Y, train_ix,valid_ix,C):    
    reg = LogisticRegression(C=C, tol=1e-10)
    reg.fit(X[train_ix],Y[train_ix])
    erroVal = 1 - reg.score(X[valid_ix],Y[valid_ix])
    erroTreino =  1 - reg.score(X[train_ix],Y[train_ix])
    return (erroTreino,erroVal)

#File Loading
def load_file(file):
    matrix = np.loadtxt(file,delimiter='\t')
    return matrix
tests = load_file("TP1_test.tsv")
train = load_file("TP1_train.tsv")

#Shuffle
tests = shuffle(tests)
train = shuffle(train)

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
X_finaltest = (X_finaltest-means)/stdevs

print("Preparing training set test/validation")
X_train,X_test,Y_train,Y_test = train_test_split(Xs, Ys, test_size=0.33, stratify = Ys)

folds = 5
stratKf = StratifiedKFold( n_splits = folds)

sep("Logistic Regression")

'''Logistic Regression Area Code '''
#Create array of C 
c_par = []
c_n = 1e-3
for x in range(16):
    c_par.append(c_n)
    c_n *=10
    
errorTrain_l = []
errorValidation_l = []

counter = 0
ind = 0
smaller = 1
cs = []

for c in c_par: 
    tr_err = va_err = 0 
    for tr_ix, val_ix in stratKf.split(Y_train, Y_train):
        r, v = calc_fold_logistic(X_train,  Y_train, tr_ix, val_ix,c)
        tr_err += r
        va_err += v
    
    cs.append(c)
    if(smaller>va_err/folds):
        smaller = va_err/folds
        ind = counter
        
    counter+=1
    errorTrain_l.append(tr_err/folds)
    errorValidation_l.append(va_err/folds)
          
print("Best of C :", cs[ind])    

plt.figure(figsize=(8,8), frameon=True)
ax_lims=(-3,3,-3,3)
plt.axis(ax_lims)
plt.subplot(211)
plt.title("Logistic Regression with best C: "+str(cs[ind]))
line1, = plt.plot(errorTrain_l, label="Train Err", linestyle='-', color='blue')
line2, = plt.plot(errorValidation_l, label="Validation Err", linestyle='-', color='green')
legend = plt.legend(handles=[line1,line2], loc='upper right')
ax = plt.gca().add_artist(legend)
plt.show()
plt.savefig('LR.png', dpi=300)
plt.close()

reg = LogisticRegression(C=cs[ind], tol=1e-10)
reg.fit(Xs, Ys)
erroVal = 1 - reg.score(X_finaltest,Y_finaltest)
'''
FAZER CONTAGEM DE ERROS e SUCESSOS
'''
print("resultado do teste erro de avaliação:",erroVal)

sep("Gaussian")

gaus = GaussianNB()
gaus.fit(Xs, Ys)
erroVal = 1 - gaus.score(X_finaltest,Y_finaltest)
print("resultado do teste erro de avaliação:",erroVal)

'''
All Code For Bayes
'''
sep("Naive Bayes")

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

errorTrain_b = []
errorValidation_b = []
best_err = 1e12
best_bw = 1
bws = [round(b,3) for b in np.arange(0.02,0.6,0.02) ]
for bandwidth in bws: 
    tr_err = va_err = 0   
    for tr_ix, val_ix in stratKf.split(Y_train, Y_train):
        r,v = calc_folds_bayes(X_train,Y_train, tr_ix,val_ix, bandwidth) 
        tr_err += 1 - accuracy_score(r , Y_train[tr_ix])
        va_err += 1 - accuracy_score(v , Y_train[val_ix])

    tr_err = tr_err/folds
    va_err = va_err/folds
    errorTrain_b.append(tr_err)
    errorValidation_b.append(va_err)  
    if va_err < best_err:
        best_err = va_err
        best_bw = bandwidth

plt.figure(figsize=(8,8), frameon=True)
ax_lims=(-3,3,-3,3)
plt.axis(ax_lims)
plt.subplot(211)
plt.title("Naive Bayes with best Bandwidth: "+str(best_bw))
line1, = plt.plot(bws,errorTrain_b, label="Train Err", linestyle='-', color='blue')
line2, = plt.plot(bws,errorValidation_b, label="Validation Err", linestyle='-', color='green')
legend = plt.legend(handles=[line1,line2], loc='lower right')
ax = plt.gca().add_artist(legend)
plt.show()
plt.savefig('NB.png', dpi=300)
plt.close()
   
r,v = bayes(Xs,Ys, X_finaltest,Y_finaltest, best_bw)
error = 1 - accuracy_score(v , Y_finaltest)
print("Best Bandwidth Found "+str(best_bw)+" with Error of",error)
'''
FAZER CONTAGEM DE ERROS e SUCESSOS
'''


