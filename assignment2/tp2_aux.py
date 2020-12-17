#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions for assignment 2
"""
import numpy as np
from skimage.io import imread
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

#imports for saving/loading
import json, codecs, os.path
from os import path
import random
#

#constants
FIGSIZE = (7,7)

# function def
def saveFeatures(data,name):
    file = name + '.json'
    dataList = data.tolist()
    json.dump(dataList, codecs.open(file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        
def loadFeatureFile(name):
    file = name + '.json'
    data = []
    if(path.exists(file)):
        loadfile = codecs.open(file, 'r', encoding='utf-8').read()
        json_raw = json.loads(loadfile)
        data = np.array(json_raw)        
    return data

def plot_hist(X,y):
    plt.title('Histogram')
    plt.figure(figsize=FIGSIZE)
    plt.hist(X, bins='auto')
    plt.show()
    
def plot_elbow(X,y,file_name="elbowplot.png"):
    plt.figure(figsize=FIGSIZE)
    plt.title('Elbow')
    f1, = plt.plot(X, color='red', label='Elbow')
    plt.legend(handles=[f1])
    plt.savefig(file_name, dpi=200, bbox_inches='tight')
    
def plot_iris(X,y,file_name="plot.png"):
    plt.figure(figsize=FIGSIZE)
    plt.plot(X[y==0,0], X[y==0,1],'o', markersize=7, color='grey', alpha=0.7)
    plt.plot(X[y==1,0], X[y==1,1],'o', markersize=7, color='orange', alpha=0.7)
    plt.plot(X[y==2,0], X[y==2,1],'o', markersize=7, color='red', alpha=0.7)
    plt.plot(X[y==3,0], X[y==3,1],'o', markersize=7, color='blue', alpha=0.7)
    plt.gca().set_aspect('equal',adjustable='box')
    plt.savefig(file_name, dpi=200, bbox_inches='tight')
    
def plot_label_classification(X,y,file_name="labelclassify.png"):
    plt.figure(figsize=FIGSIZE)
    plt.title('%d labels' % len(X))
    f1, = plt.plot(X[y==0,0], X[y==0,1],'o', markersize=7, color='grey', alpha=0.4, label='Unclassified')
    f2, = plt.plot(X[y==1,0], X[y==1,1],'o', markersize=7, color='orange', alpha=0.5, label='Phase 1')
    f3, = plt.plot(X[y==2,0], X[y==2,1],'o', markersize=7, color='red', alpha=0.5, label='Phase 2')
    f4, = plt.plot(X[y==3,0], X[y==3,1],'o', markersize=7, color='blue', alpha=0.5, label='Phase 3')
    plt.legend(handles=[f1,f2,f3,f4])
    plt.gca().set_aspect('equal',adjustable='box')
    plt.savefig(file_name, dpi=200, bbox_inches='tight')
    
def plot_centroids(X,y,centroids,file_name="centroidplot.png"):
    plt.figure(figsize=FIGSIZE)
    plt.title('Clusters & Centroids')
    plt.plot(X[y==0,0], X[y==0,1],'o', markersize=7, color='blue', alpha=0.4)
    plt.plot(X[y==1,0], X[y==1,1],'o', markersize=7, color='green', alpha=0.7)
    plt.plot(X[y==2,0], X[y==2,1],'o', markersize=7, color='red', alpha=0.7)
    plt.plot(X[y==3,0], X[y==3,1],'o', markersize=7, color='orange', alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x',
    color='k',s=100, linewidths=3)
    plt.gca().set_aspect('equal',adjustable='box')
    plt.savefig(file_name, dpi=200, bbox_inches='tight')

def plotdesci(X,file_name):
    plt.figure(figsize=FIGSIZE)
    plt.title('18 Features Plot')
    plt.xlim(xmax = 18, xmin = 0)
    plt.grid(color='black', linestyle='-', linewidth=1)
    for i in range(18):
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        f1, = plt.plot(X[i], color=color)
    
    plt.savefig(file_name, dpi=400, bbox_inches='tight')
    
def panda_plots(Features,ClassLabels,Title):
    df = pd.Dataframe(Features)
    plt(df)
    plt.title(Title)
    plt.show()

def plot_db(X,labels):
    # Black removed and is used for noise instead.
    plt.figure(figsize=FIGSIZE)
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        if k == -1: #draw noise
            xy = X[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=7, alpha=0.33)
        else: #draw class
            xy = X[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=10, alpha=0.77)
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    plt.title('DBSCAN - Estimated number of clusters: %d' % n_clusters_)
    plt.show()

def images_as_matrix(N=563):
    """
    Reads all N images in the images folder (indexed 0 through N-1)
    returns a 2D numpy array with one image per row and one pixel per column
    """
    return np.array([imread(f'images/{ix}.png',as_gray=True).ravel() for ix in range(563)])
        

def report_clusters(ids, labels, report_file):
    """Generates html with cluster report
    ids is a 1D array with the id numbers of the images in the images/ folder
    labels is a 1D array with the corresponding cluster labels
    """
    diff_lbls = list(np.unique(labels))
    diff_lbls.sort()
    html = ["""<!DOCTYPE html>
    <html lang="en">
       <head>
        <meta http-equiv="content-type" content="text/html; charset=utf-8">
        <meta charset="UTF-8">
        <title>Cluster Report</title>
       </head>
       <body>
       """]
    for lbl in diff_lbls:
        html.append(f"<h1>Cluster {lbl}</h1>")        
        lbl_imgs = ids[labels==lbl]          
        for count,img in enumerate(lbl_imgs):                
            html.append(f'<img src="images/{int(img)}.png" />')
            #if count % 10 == 9:
            #    html.append('<br/>')
    html.append("</body></html>")   
    with open(report_file,'w') as ofil:
        ofil.write('\n'.join(html))

DIV_STYLE = """style = "display: block;border-style: solid; border-width: 5px;border-color:blue;padding:5px;margin:5px;" """

#From Scilearn - DBSCAN Demo
def DBSCAN_Report(X,labels,labels_true):
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

def cluster_div(prev,ids,lbl_lists):
    div = []    
    lbls = [lbl[0] for lbl in lbl_lists]
    lbls = list(np.unique(lbls))
    lbls.sort()
    for lbl in lbls:
        div.append(f'<div {DIV_STYLE}>\n<h1>Cluster{prev}{lbl}</h1>')        
        indexes = [ix for ix in range(len(ids)) if lbl_lists[ix][0]==lbl]
        current_indexes = [ix for ix in indexes if len(lbl_lists[ix]) == 1]
        next_indexes = [ix for ix in indexes if len(lbl_lists[ix]) > 1]
        for ix in current_indexes:
                div.append(f'<img src="images/{int(ids[ix])}.png" />')
        if len(next_indexes)>0:            
            #print(f'**{prev}**\n',indexes,'\n  ',current_indexes,'\n   ',next_indexes, len(next_indexes))        
            next_ids = [ids[ix] for ix in next_indexes]
            next_lbl_lists = [lbl_lists[ix][1:] for ix in next_indexes]
            #print('****',next_lbl_lists)
            div.append(cluster_div(f'{prev}{lbl}-',next_ids,next_lbl_lists))
        div.append('</div>')
    return '\n'.join(div)
    

def report_clusters_hierarchical(ixs,label_lists,report_file):
    html = ["""<!DOCTYPE html>
    <html lang="en">
       <head>
        <meta http-equiv="content-type" content="text/html; charset=utf-8">
        <meta charset="UTF-8">
        <title>Cluster Report</title>
       </head>
       <body>
       """]   
    html.append(cluster_div('',ixs,label_lists))   
    html.append("</body></html>")   
    with open(report_file,'w') as ofil:
        ofil.write('\n'.join(html))