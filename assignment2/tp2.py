'''
Assignment 2 by
Alexander Denisov (44592)
Samuel Robalo (41936)
AA 20/21
TP4 Instructor: Joaquim Francisco Ferreira da Silva
Regency: Ludwig Krippahl
'''
'''
Goal: 
The goal of this assignment is to examine a set of bacterial cell images 
using machine learning techniques, including feature extraction, 
features selection and clustering, in order to help the biologists organize similar images.
'''
'''
Resources: 
A set of 563 PNG images (in the images/ folder) 
taken from a super-resolution fluorescence microscopy photograph of Staphylococcus aureus.
All images have the same dimensions, 50 by 50 pixels, 
with a black background and the segmented region centered in the image. 
'''
'''
In this assignment, you will load all images, extract features, 
examine them and select a subset for clustering with the goal of 
reaching some conclusion about the best way of grouping these images.
'''

#imports
import tp2_aux as aux 
import numpy as np
from collections import Counter
#

DECOMP_NUM_FEATURES = 6
NUM_IMAGES = 563
FEATURE_LBL = []

'''FeatureLabeling'''
def featureLabel():
    x = 1
    while x < 17:
        FEATURE_LBL.append('F'+str(x))
        x+=1
    #print(features)
'''End of FeatureLabeling'''

featureLabel()
imgMatrix = aux.images_as_matrix(NUM_IMAGES)
print('Info: Loaded',imgMatrix.shape[0],'images.')
input_data = "labels.txt"
cell_cycle_labels = np.loadtxt(input_data, delimiter=",")
print('Info: Loaded label information.')
#print(cell_cycle_labels) #output check

def labelReporting(labels):
    lbls = labels[:,1]
    diff_lbls = list(np.unique(lbls))
    diff_lbls.sort()
    p0 = Counter(lbls).get(0); p1 = Counter(lbls).get(1)
    p2 = Counter(lbls).get(2); p3 = Counter(lbls).get(3)
    print('There are',str(len(diff_lbls)),'different labels.\n', diff_lbls)
    print('Total classified:',str(p1+p2+p3))
    print('',p1,'labeled with Phase 1.\n',p2,'labeled with Phase 2.\n',
          p3,'labeled with Phase 3.')
    print('Total unclassified:',str(p0),'\n')
    
labelReporting(cell_cycle_labels)

'''
From this matrix, you will extract features using three different methods:

Principal Component Analysis (PCA)
    >Use the PCA class from the sklearn.decomposition module.
t-Distributed Stochastic Neighbor Embedding (t-SNE)
    >Use the TSNE class from the sklearn.manifold module. When creating an object of this class, use the method='exact' argument, for otherwise the TSNE constructor will use a faster, approximate, computation which allows for at most 3 components.
Isometric mapping with Isomap
    >Use the Isomap class from the sklearn.manifold module.
    
With each method, extract six features from the data set, for a total of 18 features.
'''

###Start of Feature Extraction
#Feature extraction is the process of computing features from the initial data

#imports
import sklearn.decomposition as decomp
import sklearn.manifold as manifold
#

#PreProcess - Standard Scale
import sklearn.preprocessing as preprocess
stand_scale =  preprocess.StandardScaler() 
'''Question: Should we standarize the data?'''
std_answer = False
#In unsupervised learning, we often need to be careful about
#how we transform the data because the shape of its distribution 
#and the distances between the points may be important.
if(std_answer):
    X = stand_scale.fit_transform(imgMatrix)
else:
    X = imgMatrix

allowDecomposition = True

if(allowDecomposition):
    print('[Feature Extraction]\nExtracting '+str(DECOMP_NUM_FEATURES)+' features for each method.')
    
    #PCA Feature Extraction
    pca = decomp.PCA(n_components=DECOMP_NUM_FEATURES)
    X_pca = pca.fit_transform(X)
    print('(1/3) PCA Complete')
    #print(X_pca.shape) #output check
    #print(X_pca) #output check
    
    #Isometric Feature Extraction
    isom = manifold.Isomap(n_components=DECOMP_NUM_FEATURES)
    X_isom = isom.fit_transform(X)
    print('(2/3) Isometric Complete')
    #print(X_isom.shape) #output check
    #print(X_isom) #output check
    
    #t-SNE Feature Extraction
    tsne = manifold.TSNE(n_components=DECOMP_NUM_FEATURES, method='exact')
    X_tsne = tsne.fit_transform(X)
    print('(3/3) t-SNE Complete')
    #print(X_tsne.shape) #output check
    #print(X_tsne) #output check
    
    print('[End of Feature Extraction]')
###End of Feature Extraction

print('[End of Execution]')

'''
(Selecting Best Features after extraction)

Nem há um valor fixo para o número de features. 
Terá que ser determinado pelas experiências. 
Por ex: se o valor de F que vem que f_classf nos der 
valores de por ex: 60, 40, 2, 0.5, 0.25, 0.002 
isto significa que o número de features
a considerar neste critério devem ser 2. 
Esta é uma pista que deve ser confirmada com gráficos 
em que os eixos são pares de features, 
mostrando quão bom ou não é o poder discriminate delas para separar as classes.

É a partir das 16 que se extraem as melhores.  
Mas é improvável que venhamos a usar as 16 
por que nem todas são realmente discriminates. 
Eu diria um número não superior a 4 ou 5 é tipicamente o usado. 
É preciso experimentar. 
O trabalho tem essa componente de experimentação.
'''

#Best Features


#Clustering

'''
Posteriormente ao clustering, 
também poderemos chegar à conclusão de 
qual o melhor grupo candidato de features.
'''

#Clustering Best Group of Features