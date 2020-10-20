'''
Auxiliary methods from classes & others
'''

import math
import numpy as np
import matplotlib.pyplot as plt

def aproxNormalTest(X,N,P) -> float:
    """Aprox Normal Distribution
    @params:
        X - Required : measured number of errors (Int)
        N - Required : size of test set (Int)
        P - Required : expected number of errors (Int)  
    @return: 
        Z   - aprox normal distribution (float)"""
    z = (X-N*P)/(math.sqrt(N*P*(1-P)))
    return z

def McNemarTest(e01,e10) -> float:
    """Value of Estatisticly Diferent Mistakes done by 2 classifiers
        with 95% confidence level of 3.84
    @params:
        e01 - Required : number of examples this classifer got wrong (Int)
        e10 - Required : number of examples this classifer got wrong (Int)
    @return: 
        X   - value"""
    X = ((abs(e01-e10)-1)**2)/(e01+e10)
    print(str("Classifier is likely better if "+X+" >= 3.84"))
    return X