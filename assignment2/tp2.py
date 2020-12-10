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

DECOMP_NUM_FEATURES = 6
NUM_IMAGES = 563

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
#In unsupervised learning, we often need to be careful about
#how we transform the data because the shape of its distribution 
#and the distances between the points may be important.
X = imgMatrix
allowFeatureProcessing = True

X_pca = []
X_isom = []
X_tsne = []
X_features = []

X_pca = aux.loadFeatureFile('pca')
X_isom = aux.loadFeatureFile('isom')
X_tsne = aux.loadFeatureFile('tsne')

if(len(X_pca)>0 and len(X_isom)>0 and len(X_tsne)>0):
    allowFeatureProcessing = False

if(allowFeatureProcessing):
    print('[Feature Extraction]\nExtracting '+str(DECOMP_NUM_FEATURES)+' features for each method.')
    
    #PCA Feature Extraction
    pca = decomp.PCA(n_components=DECOMP_NUM_FEATURES)
    X_std_pca = stand_scale.fit_transform(X)
    X_pca = pca.fit_transform(X_std_pca)
    print('(1/3) PCA Complete')
    #print(X_pca.shape) #output check
    #print(X_pca) #output check
    
    #Isometric Feature Extraction
    isom = manifold.Isomap(n_components=DECOMP_NUM_FEATURES)
    X_std_isom = stand_scale.fit_transform(X)
    X_isom = isom.fit_transform(X_std_isom)
    print('(2/3) Isometric Complete')
    #print(X_isom.shape) #output check
    #print(X_isom) #output check
    
    #t-SNE Feature Extraction
    tsne = manifold.TSNE(n_components=DECOMP_NUM_FEATURES, method='exact')
    X_std_tsne = stand_scale.fit_transform(X)
    X_tsne = tsne.fit_transform(X_std_tsne)
    print('(3/3) t-SNE Complete')
    #print(X_tsne.shape) #output check
    #print(X_tsne) #output check

    aux.saveFeatures(X_pca,'pca')
    aux.saveFeatures(X_isom,'isom')
    aux.saveFeatures(X_tsne,'tsne')
    
    print('[End of Feature Extraction]')
###End of Feature Extraction

X_features = np.append(X_features,X_pca.T)
X_features = np.append(X_features,X_isom.T)
X_features = np.append(X_features,X_tsne.T)
X_features = X_features.reshape(DECOMP_NUM_FEATURES*3,NUM_IMAGES).T

'''
i=0
for i in range (6):
    if(X_features[0][i] == X_pca[0][i]):
        print("PCA Checks Out!")
    if(X_features[0][i+6] == X_isom[0][i]):
        print("ISOM Checks Out!")
    if(X_features[0][i+12] == X_tsne[0][i]):
        print("TSNE Checks Out!")
'''
labelledCells = cell_cycle_labels[cell_cycle_labels[:,1] != 0,:]
y=np.array(labelledCells[:,1])

labelledFeatures = X_features[cell_cycle_labels[:,1] != 0,:]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

sample = f_classif(labelledFeatures, y)
print('ANOVA Classification [F-value,p-value]')
ANOVAValues = np.array(sample).T
print(ANOVAValues)

nbestcounter = 0
index = 0
bestFeaturesIndex = []

for nbestcounter in range(2):
    for index in range(len(ANOVAValues)):
        value = ANOVAValues[index][0]
        if(value == ANOVAValues.max() and (index not in bestFeaturesIndex)):
            ANOVAValues[index][0] = 9e-99
            bestFeaturesIndex.append(index)

print('BestFeaturesIndex:',bestFeaturesIndex)
print('X_18Features Shape:', X_features.shape)
X_4features = []
nbestcounter = 0
for nbestcounter in range(4):
    X_4features.append(X_features.T[:][bestFeaturesIndex[nbestcounter]])

X_4features = np.array(X_4features).T

print('X_4Features Shape:',X_4features.shape)
#print('X_4Features:',X_4features)
#print('X_Features:',X_features)

ovr = OneVsRestClassifier(SVC(kernel='rbf', gamma=0.7, C=10)).fit(labelledFeatures, y)
prediction = ovr.predict(sample)
print("OneVsRestClassifier")
print(prediction)

# Create an SelectKBest object to select features with two best F-Values
print("SelectKBest Features")
fvalue_selector = SelectKBest(f_classif, k=5)
# Apply the SelectKBest object to the features and target
# Selecionado as que tem menos probabilidade e maior F1-score (probabilidade de independencia dos dados ser maior)
X_kbest = fvalue_selector.fit_transform(labelledFeatures, y)

#plot
#aux.plot_iris(X_kbest, y)

from sklearn.neighbors import KNeighborsClassifier


ones = np.ones(cell_cycle_labels[:,1].shape[0])
dist,index = KNeighborsClassifier(n_neighbors=5).fit(X_4features, ones).kneighbors()
classifff = np.amax(dist,1)
classifff[::-1].sort()

deriv = [ (classifff[i]-classifff[i-1])/(1/float(classifff.shape[0]) )  for i in range(1,classifff.shape[0])]

#aux.plot_elbow(classifff/classifff[0],index[:,1])

aux.plot_elbow(classifff,index[:,1])

from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=10.1, min_samples=5)
model=dbscan.fit(X_4features)
labelsdb = model.fit_predict(X_4features)
aux.DBSCAN_Report(X_4features,labelsdb,cell_cycle_labels[:,1])

kmeans = KMeans(n_clusters=3).fit(X_4features)
labelskm = kmeans.predict(X_4features)
centroids = kmeans.cluster_centers_

#aux.plot_label_classification(X_features, cell_cycle_labels[:,1])

aux.plot_centroids(X_4features, labelskm, centroids, file_name='centroid.png')

aux.plot_db(X_4features,labelsdb)

aux.report_clusters(cell_cycle_labels[:,0], labelsdb ,"cluster_report.html")


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

É a partir das 18 que se extraem as melhores.  
Mas é improvável que venhamos a usar as 18 
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