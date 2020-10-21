'''
Auxiliary methods from classes & others
'''

import math
import numpy as np

def aproxNormalTest(X,N,P) -> float:
    '''Aprox Normal Distribution
    @params:
        X - Required : measured number of errors (Int)
        N - Required : size of test set (Int)
        P - Required : expected number of errors (Int)  
    @return: 
        Z   - aprox normal distribution (float)'''
    z = (X-N*P)/(math.sqrt(N*P*(1-P)))
    return z

def McNemarTest(e01,e10) -> float:
    '''Value of Estatisticly Diferent Mistakes done by 2 classifiers
        with 95% confidence level of 3.84        
    @params:
        e01 - Required : number of examples this classifer got wrong (Int)
        e10 - Required : number of examples this classifer got wrong (Int)        
    @return: 
        X   - value'''
    X = ((abs(e01-e10)-1)**2)/(e01+e10)
    print(str("[McNemar's Test'] Classifier is likely better if "+X+" >= 3.84"))
    return X

def GaussianKernelDensity(bandwidth,features):
    '''Obtain list of (features_0,features_1)
    @params:
        bandwidth - Required : number of examples this classifer got wrong (Int)
        features - Required : number of examples this classifer got wrong (Int)        
    @return: 
        Array (features_0,features_1)'''
    ktype = "gaussian"
    feat_0 = []
    feat_1 = []
    print(features)
    for feature in features:
        feat_0.append(KernelDensity(kernel=ktype, bandwidth=bandwidth).fit(t_0[:,i]).reshape(-1,1))
        feat_1.append(KernelDensity(kernel=ktype, bandwidth=bandwidth).fit(t_1[:,i]).reshape(-1,1))
    print (feat_0,feat_1)
    return (feat_0,feat_1)
        
# We can estimate to joint probability distribution
# simply by the product of the prior probablility of belonging to the class
# and the product of the probablility of having each individual feature 
# with that particular value if it belong to the class
def NBC(train_set, features, test_set):
    '''
    Naive Bayes Classifier
    '''
    
    _NBC_Train_Aux()
    
    _NBC_Classify_Aux()
    
    return 0

def _NBC_Train_Aux():
    '''
    Private NBC Train
    '''
    #>>Get prior probablility of belonging to the class
    #>it is simply the fraction of examples that belong to that class
    # ln p(Ck)
    
    prior_prob = sum(d)/len(d)
    log_prior_prob= math.log(prior_prob)
    
    feature_values = []
    for k in features: #for each feature
        feature_values.append( log_prior_prob + Sum ln p(xj|Ck) )
        
    best_feature = np.argmax(feature_values)

    np.argmax("Best feature: "+str(best_feature))
    
    
    #math.log( )
    #np.argmax( )
    
    return 0

def _NBC_Classify_Aux():
    '''
    Private NBC Classify
    '''
    return 0