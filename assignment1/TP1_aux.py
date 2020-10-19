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
    