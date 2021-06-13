# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 00:10:02 2021

@author: stavros moutsis
"""

#    LIBRARIES 

import cv2
import numpy as np
import math
import pywt
from sklearn.cluster import KMeans
from os.path import isfile, join
from os import listdir
import os
from pathlib import Path
import misvm
from sklearn import   metrics, model_selection
import scipy.spatial.distance as dist
from CitationKNN import CitationKNN

'''   From RGB color model to YCbCr color model, used by k-meansSeg !!!!!!!!!!!!!!!!!!!!!!! '''
####  From RGB color model to YCbCr color model, used my k-means-Seg !!!!!!!!!!!!!!!!!!!!!!!
def rgb2ycbcr(im):
    cbcr = np.empty_like(im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    
    # # Y
    # cbcr[:,:,0] = .299 * r + .587 * g + .114 * b
    # # Cb
    # cbcr[:,:,1] = 128 - .169 * r - .331 * g + .5 * b
    # # Cr
    # cbcr[:,:,2] = 128 + .5 * r - .419 * g - .081 * b
    
    # Y
    cbcr[:,:,0] = .299 * r + .587 * g + .114 * b
    # Cb
    cbcr[:,:,1] = 128 - .1687 * r - .3313 * g + .5 * b
    # Cr
    cbcr[:,:,2] = 128 + .5 * r - .4197 * g - .0813 * b
    
    return np.uint8(cbcr)

def distance(x,y):
    
    
    temp = np.zeros(len(y))
    for p in range(len(y)):
        dist1 = 0
        for t in range(6):
            
            dist1 = (pow(abs(x[t]-y[p][t]),2)) + dist1
        # temp[p] = math.sqrt(dist1)
        temp[p] = (dist1)

    return temp

'''   generator3 func = k-meansSeg   Bag generator !!!!!!!!!!!!!!!!!!!! '''
####  generator3 func = k-meansSeg   Bag generator !!!!!!!!!!!!!!!!!!!!
def generator3(image2):
    
    image = np.array(image2)

    imSize = np.shape(image)


    thresh_k = 16
    blobSize = np.array([[4],[4]])
    blobSize = np.reshape(blobSize,(1,2))

    tempSize = np.shape( blobSize );

    if tempSize[1] == 1:
        blobRowSize = blobSize
        blobColSize = blobSize
    else:
        blobRowSize = blobSize[0][0]
        blobColSize = blobSize[0][1]



    blobsRow = math.floor( imSize[0] / blobRowSize )
    blobsCol = math.floor( imSize[1] / blobColSize )

    if blobsRow == 0:
        blobsRow = 1;

    if blobsCol == 0:
        blobsCol = 1;


    ycbcr = rgb2ycbcr(image)

    Y = ycbcr[:,:,0]
    Cb = ycbcr[:,:,1]
    Cr = ycbcr[:,:,2]

    # cv2.imshow('original',image)
    # cv2.imshow('img2',ycbcr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    tmpbag = np.zeros((blobsRow * blobsCol, 6))
    # blob_map = np.zeros(np.shape(Y))

    blob_idx = 0

    tmpbag = np.zeros((blobsCol*blobsRow,6))
    

    for i in range(blobsRow):
        for j in range(blobsCol):
            # idx1 = np.reshape(np.arange((i*blobRowSize+1),(min(imSize[0]-1,(i+1)*blobRowSize)+1),dtype = float),(1,(min(imSize[0]-1,(i+1)*blobRowSize)-(i*blobRowSize+1))+1))
            # idx2 = np.reshape(np.arange((i*blobColSize+1),(min(imSize[1]-1,(i+1)*blobColSize)+1),dtype = float),(1,(min(imSize[1]-1,(i+1)*blobColSize)-(i*blobColSize+1))+1))
        
            idx1_1 = i*blobRowSize
            idx1_2 = idx1_1+4
        
            idx2_1 = j*blobColSize
            idx2_2 = idx2_1+4
        
        
            tmpbag[blob_idx][0] = np.mean(np.mean(Y[idx1_1:idx1_2,idx2_1:idx2_2],axis=0))
            tmpbag[blob_idx][1] = np.mean(np.mean(Cb[idx1_1:idx1_2,idx2_1:idx2_2],axis=0))
            tmpbag[blob_idx][2] = np.mean(np.mean(Cr[idx1_1:idx1_2,idx2_1:idx2_2],axis=0))
        
            coeffs = pywt.dwt2(Y[idx1_1:idx1_2,idx2_1:idx2_2], 'db4')
            cA, (cH, cV, cD) = coeffs
        
            tmpbag[blob_idx][3] = pow(np.mean(np.mean(np.power(cH,2),axis=0)),0.5)
            tmpbag[blob_idx][4] = pow(np.mean(np.mean(np.power(cV,2),axis=0)),0.5)
            tmpbag[blob_idx][5] = pow(np.mean(np.mean(np.power(cD,2),axis=0)),0.5)
        
            blob_idx = blob_idx + 1
        

    thresh_D = 1e5
    thresh_der = 1e-12

    # allD = np.reshape(np.zeros(thresh_k*thresh_k),(thresh_k,thresh_k))
    # allD[0][0] = 1e20

    allD = np.zeros(thresh_k+1)
    allD[0] = 1e20
    allD[1] = 1e20


    for k in range(2,thresh_k+1):
    # for k in range(2,3):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(tmpbag)
    
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        sumd = np.reshape(np.zeros(blob_idx*k),(blob_idx,k))
        sumd2 = np.zeros(blob_idx)
    
   
        for j in range(blob_idx):
            sumd[j] =  distance(tmpbag[j],centroids)        
            sumd2[j] = min(sumd[j])*min(sumd[j])
    
        allD[k] = sum(sumd2)
    
        if (allD[k]<thresh_D) or ( k>=3 and (allD[k]-allD[k-1])/(allD[3] - allD[1])/2 < thresh_der  ):
            break
    
    
    
    bag = np.reshape(np.zeros(k*6, dtype=float),(k,6))
    [row, col] = np.shape(tmpbag)

    if row == 1:
        bag = tmpbag
    else:
        for i in range(k):
            count=0
            for j in range(blobsCol*blobsRow):
            
                if labels[j] == i:
                    count = count+1
                
                    bag[i]= bag[i] + tmpbag[j]
            bag[i] = bag[i]/count
            # print(count)
            
    # minn = np.min(bag)
    # maxx = np.max(bag)

    # bag = (bag-minn)/(maxx-minn)          

    for i in range(len(bag)):
        minn = np.min(bag)
        maxx = np.max(bag)
    
        bag[i] = (bag[i]-minn)/(maxx-minn)    
  
    return bag

'''   generator2 func = ROW   Bag generator !!!!!!!!!!!!!!!!!!!! '''
####  generator2 func = ROW   Bag generator !!!!!!!!!!!!!!!!!!!!
def generator2(image2):
    
    blur = cv2.GaussianBlur(image2,(5,5),0)
    image = np.array(blur)

    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # cv2.imshow('image2', blur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    resize = 15

    resizedImage = cv2.resize(image, (resize,resize), interpolation = cv2.INTER_AREA)

    newSize = resizedImage.shape

    rowSumRGB = np.reshape(np.zeros(newSize[0]*len(newSize)),(newSize[0],len(newSize)))
    # cv2.imshow('image2', resizedImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for i in range(newSize[0]):  
    
        rowSumRGB[i][0] = sum((resizedImage[i])[:,2])
        rowSumRGB[i][1] = sum((resizedImage[i])[:,1]) 
        rowSumRGB[i][2] = sum((resizedImage[i])[:,0])
    

    rowMeanRGB = rowSumRGB / newSize[1];

    bag = np.reshape(np.zeros(newSize[0]*9), (newSize[0],9))

    i=0

    bag[i,0] = rowMeanRGB[i,0]
    bag[i,1] = rowMeanRGB[i,1]
    bag[i,2] = rowMeanRGB[i,2]

    bag[i,3] = rowMeanRGB[i,0] - rowMeanRGB[newSize[0]-1,0]
    bag[i,4] = rowMeanRGB[i,1] - rowMeanRGB[newSize[0]-1,1]
    bag[i,5] = rowMeanRGB[i,2] - rowMeanRGB[newSize[0]-1,2]

    bag[i,6] = rowMeanRGB[i,0] - rowMeanRGB[1,0]
    bag[i,7] = rowMeanRGB[i,1] - rowMeanRGB[1,1]
    bag[i,8] = rowMeanRGB[i,2] - rowMeanRGB[1,2]
 
    for i in range(1,newSize[0]-1):
    
        bag[i,0] = rowMeanRGB[i,0]
        bag[i,1] = rowMeanRGB[i,1]
        bag[i,2] = rowMeanRGB[i,2]

        bag[i,3] = rowMeanRGB[i,0] - rowMeanRGB[i-1,0]
        bag[i,4] = rowMeanRGB[i,1] - rowMeanRGB[i-1,1]
        bag[i,5] = rowMeanRGB[i,2] - rowMeanRGB[i-1,2]

        bag[i,6] = rowMeanRGB[i,0] - rowMeanRGB[i+1,0]
        bag[i,7] = rowMeanRGB[i,1] - rowMeanRGB[i+1,1]
        bag[i,8] = rowMeanRGB[i,2] - rowMeanRGB[i+1,2]


    i = newSize[0]-1

    bag[i,0] = rowMeanRGB[i,0]
    bag[i,1] = rowMeanRGB[i,1]
    bag[i,2] = rowMeanRGB[i,2]

    bag[i,3] = rowMeanRGB[i,0] - rowMeanRGB[i-1,0]
    bag[i,4] = rowMeanRGB[i,1] - rowMeanRGB[i-1,0]
    bag[i,5] = rowMeanRGB[i,2] - rowMeanRGB[i-1,0]

    bag[i,6] = rowMeanRGB[i,0] - rowMeanRGB[0,0]
    bag[i,7] = rowMeanRGB[i,1] - rowMeanRGB[0,1]
    bag[i,8] = rowMeanRGB[i,2] - rowMeanRGB[0,2]


    for i in range(len(bag)):
        minn = np.min(bag)
        maxx = np.max(bag)
    
        bag[i] = (bag[i]-minn)/(maxx-minn)
        
        
    return bag

'''  generator func = SBN   Bag generator !!!!!!!!!!!!!!!!!!!! '''
#### generator func = SBN   Bag generator !!!!!!!!!!!!!!!!!!!!
def generator(image2):
    
    # image = mpimg.imread(filepath)
    blur = cv2.GaussianBlur(image2,(5,5),0)
    image = np.array(blur)

    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # cv2.imshow('image2', blur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    resize = 8

    resizedImage2 = cv2.resize(image, (resize,resize), interpolation = cv2.INTER_AREA)
    resizedImage = cv2.resize(image, (resize,resize), interpolation = cv2.INTER_AREA)
    # temp = resizedImage
    resizedImage[:,:,0:1] = resizedImage2[:,:,2:3]
    resizedImage[:,:,2:3] = resizedImage2[:,:,0:1]
    
    newSize = resizedImage.shape
    
    blobSumRGB = np.reshape(np.zeros((newSize[0]-1)*(newSize[1]-1)*3),((newSize[0]-1),(newSize[1]-1),3))
    
    resizedImage = resizedImage.astype(float)

    for i in range(newSize[0]-1):
        for j in range(newSize[1]-1):
            blobSumRGB[(i):(i+1), (j):(j+1), 0:1] = float(resizedImage[(i):(i+1), (j):(j+1), 0:1]) + float(resizedImage[(i+1):(i+1+1), (j):(j+1), 0:1]) + float(resizedImage[(i):(i+1), (j+1):(j+1+1), 0:1]) + float(resizedImage[(i+1):(i+1+1), (j+1):(j+1+1), 0:1])                                   
            blobSumRGB[(i):(i+1), (j):(j+1), 1:2] = float(resizedImage[(i):(i+1), (j):(j+1), 1:2]) + float(resizedImage[(i+1):(i+1+1), (j):(j+1), 1:2]) + float(resizedImage[(i):(i+1), (j+1):(j+1+1), 1:2]) + float(resizedImage[(i+1):(i+1+1), (j+1):(j+1+1), 1:2])   
            blobSumRGB[(i):(i+1), (j):(j+1), 2:3] = float(resizedImage[(i):(i+1), (j):(j+1), 2:3]) + float(resizedImage[(i+1):(i+1+1), (j):(j+1), 2:3]) + float(resizedImage[(i):(i+1), (j+1):(j+1+1), 2:3]) + float(resizedImage[(i+1):(i+1+1), (j+1):(j+1+1), 2:3])   


    blobMeanRGB = blobSumRGB / 4

    bag = np.reshape(np.zeros((newSize[0]-5)*(newSize[1]-5)*15),((newSize[0]-5)*(newSize[1]-5),15))
    bag = bag.astype(float)

    for i in range(newSize[0]-5):
        for j in range(newSize[1]-5):
       
            bag[((newSize[0]-5)*i+j):((newSize[0]-5)*i+j+1), (0):(1)] = float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+2):(j+3),(0):(1)]) 
            bag[((newSize[0]-5)*i+j):((newSize[0]-5)*i+j+1), (1):(2)] = float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+2):(j+3),(1):(2)])
            bag[((newSize[0]-5)*i+j):((newSize[0]-5)*i+j+1), (2):(3)] = float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+2):(j+3),(2):(3)])  

            bag[((newSize[0]-5)*i+j):((newSize[0]-5)*i+j+1), (3):(4)] = float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j):(j+1),(0):(1)]) - float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+2):(j+3),(0):(1)]) 
            bag[((newSize[0]-5)*i+j):((newSize[0]-5)*i+j+1), (4):(5)] = float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j):(j+1),(1):(2)]) - float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+2):(j+3),(1):(2)])
            bag[((newSize[0]-5)*i+j):((newSize[0]-5)*i+j+1), (5):(6)] = float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j):(j+1),(2):(3)]) - float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+2):(j+3),(2):(3)])                                                        
  
            bag[((newSize[0]-5)*i+j):((newSize[0]-5)*i+j+1), (6):(7)] = float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+4):(j+5),(0):(1)]) - float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+2):(j+3),(0):(1)]) 
            bag[((newSize[0]-5)*i+j):((newSize[0]-5)*i+j+1), (7):(8)] = float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+4):(j+5),(1):(2)]) - float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+2):(j+3),(1):(2)])
            bag[((newSize[0]-5)*i+j):((newSize[0]-5)*i+j+1), (8):(9)] = float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+4):(j+5),(2):(3)]) - float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+2):(j+3),(2):(3)])    
        
            bag[((newSize[0]-5)*i+j):((newSize[0]-5)*i+j+1), (9):(10)] = float(blobMeanRGB[(newSize[0]-5+i-3):(newSize[0]-5+i-2),(j+2):(j+3),(0):(1)]) - float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+2):(j+3),(0):(1)]) 
            bag[((newSize[0]-5)*i+j):((newSize[0]-5)*i+j+1), (10):(11)] = float(blobMeanRGB[(newSize[0]-5+i-3):(newSize[0]-5+i-2),(j+2):(j+3),(1):(2)]) - float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+2):(j+3),(1):(2)])
            bag[((newSize[0]-5)*i+j):((newSize[0]-5)*i+j+1), (11):(12)] = float(blobMeanRGB[(newSize[0]-5+i-3):(newSize[0]-5+i-2),(j+2):(j+3),(2):(3)]) - float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+2):(j+3),(2):(3)])    

            bag[((newSize[0]-5)*i+j):((newSize[0]-5)*i+j+1), (12):(13)] = float(blobMeanRGB[(newSize[0]-5+i+1):(newSize[0]-5+i+2),(j+2):(j+3),(0):(1)]) - float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+2):(j+3),(0):(1)]) 
            bag[((newSize[0]-5)*i+j):((newSize[0]-5)*i+j+1), (13):(14)] = float(blobMeanRGB[(newSize[0]-5+i+1):(newSize[0]-5+i+2),(j+2):(j+3),(1):(2)]) - float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+2):(j+3),(1):(2)])
            bag[((newSize[0]-5)*i+j):((newSize[0]-5)*i+j+1), (14):(15)] = float(blobMeanRGB[(newSize[0]-5+i+1):(newSize[0]-5+i+2),(j+2):(j+3),(2):(3)]) - float(blobMeanRGB[(newSize[0]-5+i-1):(newSize[0]-5+i),(j+2):(j+3),(2):(3)])    


    for i in range(len(bag)):
        minn = np.min(bag)
        maxx = np.max(bag)
    
        bag[i] = (bag[i]-minn)/(maxx-minn)    
        
        
    return bag



c = int(input("Please if you want to run the first dataset give 1, otherwise give 2 for the second dataset:   "))
g = int(input("Please give 1 for the NBF generator, give 2 for the ROW generator, give 3 for the k-meansSeg generator:   "))


if c == 1:
    
    '''the path of the first dataset '''
    #  the path of the first dataset
    path = (r'C:\Users\Thanasis\Desktop\project_advance_topics_in_ML\miml-image-data\original')
   
    y = np.zeros(2000)
    y[0:400] = 0
    y[400:800] = 1
    y[800:1200] = 2
    y[1200:1600] = 3
    y[1600:2000] = 4
    
    onlyfiles2 = [ f for f in listdir(path) if isfile(join(path,f)) ]
    onlyfiles = [] 
    XX = []
    
    for n in range(0, (len(onlyfiles2))):
        jpg = '.jpg'
        string = str(n+1)
        onlyfiles.append(string+jpg)
    
    for i in range(2000):
        filepath = path + '\\' + onlyfiles[i]
        # filepath = Path(filepath)
        image = cv2.imread(filepath)
        if g == 3:
            image= cv2.resize(image, (100, 100))
            
        if g == 1:
            bag = generator(image)
        elif g == 2:
            bag = generator2(image)
        elif g == 3:
            bag = generator3(image)
        else:
            print("\nError: You gave wrong Input !!!\n")
            exit()
        
        
        XX.append(bag)
        print(i+1)
    
elif c == 2:
    
    '''the path of the second dataset '''
    #  the path of the second dataset
    path = (r'C:\Users\Thanasis\Desktop\project_advance_topics_in_ML\new_data\dataset2')
    
    y = np.zeros(1123)
    y[0:300] = 0
    y[300:514] = 1
    y[514:766] = 2
    y[766:1123] = 3
    
    onlyfiles2 = [ f for f in listdir(path) if isfile(join(path,f)) ]
    onlyfiles = [] 
    XX = []

    for i in range(1123):
        filepath = path + '\\' + onlyfiles2[i]
        # filepath = Path(filepath)
        image = cv2.imread(filepath)
        if g == 3:
            image= cv2.resize(image, (100, 100))
        
        if g == 1:
            bag = generator(image)
        elif g == 2:
            bag = generator2(image)
        elif g == 3:
            bag = generator3(image)
        else:
            print("\nError: You gave wrong Input !!!\n")
            exit()
        
        XX.append(bag)
        print(i+1)
        
else:
    print("\nError: You gave wrong Input !!!\n")
    exit()
    
if g == 1:
    dimensions = 15
elif g == 2:
    dimensions = 9
elif g == 3:
    dimensions = 6
else:
    print("\nError: You gave wrong Input !!!\n")
    exit()
    

X_train, X_test, y_train, y_test = model_selection.train_test_split(XX, y, test_size = 0.25, random_state = 0)

'''  START OF THE TRANSFORMATION OF DATA WITH KMEANS METHOD !!!!!!!!!! '''
#### START OF THE TRANSFORMATION OF DATA WITH KMEANS METHOD !!!!!!!!!!
 
count = 0 
for i in range(len(X_train)):
    # print(i)
    count= count + len(X_train[i])
    
X_train2 = np.reshape(np.zeros(dimensions*count), (count,dimensions))
bag_n = np.reshape(np.zeros(count), (count,1))

k=0
for i in range(len(X_train)):
    
    for j in range(len(X_train[i])):
        
        X_train2[k] = X_train[i][j]
        bag_n[k] = i
        k =k+1
n_clusters = int(input("Please give the number of clusters to form as well as the number of centroids to generate\nWe recommend 150\nn_clusters=    "))        
# n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_train2)
labels = kmeans.labels_

X_train3 = np.reshape(np.zeros(n_clusters*len(X_train)),(len(X_train),n_clusters))


prev = 0 
for i in range(count):
    
    if bag_n[i] == prev:
        for j in range(n_clusters):
            if labels[i] == j:
                X_train3[prev][j] = X_train3[prev][j] + 1
                   
    else:
        prev = int(bag_n[i][0])
        for j in range(n_clusters):
            if labels[i] == j:
                X_train3[prev][j] = X_train3[prev][j] + 1
      


count = 0 
for i in range(len(X_test)):
    count= count + len(X_test[i])
    
X_test2 = np.reshape(np.zeros(dimensions*count), (count,dimensions))
bag_n2 = np.reshape(np.zeros(count), (count,1))

k=0
for i in range(len(X_test)):
    
    for j in range(len(X_test[i])):
        
        X_test2[k] = X_test[i][j]
        bag_n2[k] = i
        k =k+1


labels2 = kmeans.predict(X_test2)
X_test3 = np.reshape(np.zeros(n_clusters*len(X_test)),(len(X_test),n_clusters))


prev = 0 
for i in range(count):
    
    if bag_n2[i] == prev:
        for j in range(n_clusters):
            if labels2[i] == j:
                X_test3[prev][j] = X_test3[prev][j] + 1
                   
    else:
        prev = int(bag_n2[i][0])
        for j in range(n_clusters):
            if labels2[i] == j:
                X_test3[prev][j] = X_test3[prev][j] + 1
   
'''  END OF THE TRANSFORMATION OF DATA WITH KMEANS METHOD !!!!!!!!!! '''  
#### END OF THE TRANSFORMATION OF DATA WITH KMEANS METHOD !!!!!!!!!! 

'''  NORMALIZATION OF NEW DATA !!!!!!!!!! '''
#### NORMALIZATION OF NEW DATA !!!!!!!!!!

from sklearn.preprocessing import MinMaxScaler 
  
scaler = MinMaxScaler().fit(X_train3)

X_train3 = scaler.transform(X_train3)
X_test3 = scaler.transform(X_test3)
  
    
'''       SVM MODEL ON NEW DATA '''    
#         SVM MODEL ON NEW DATA 

from sklearn import  svm

y_test = np.array(y_test, dtype = np.float)
y_train = np.array(y_train, dtype = np.float)
clf = svm.SVC(C=100.0, kernel='rbf',degree=7, gamma='scale')
clf.fit(X_train3, y_train)

y_predicted4 = clf.predict(X_test3)

y_test = np.array(y_test, dtype = np.float)
y_train = np.array(y_train, dtype = np.float)
print("Accuracy:  %9f" % metrics.accuracy_score(y_test, y_predicted4))
print("Precision:  %2f" % metrics.precision_score(y_test, y_predicted4, average='macro'))
print("Recall:  %11f" % metrics.recall_score(y_test, y_predicted4, average='macro'))
print("F1:  %15f \n" % metrics.f1_score(y_test, y_predicted4, average='macro'))


'''       KNN MODEL ON NEW DATA '''   
#         KNN MODEL ON NEW DATA 
   
from sklearn.neighbors import KNeighborsClassifier

y_test = np.array(y_test, dtype = np.float)
y_train = np.array(y_train, dtype = np.float)

KNN_model = KNeighborsClassifier(n_neighbors=50,weights='distance', p=1)
KNN_model.fit(X_train3,y_train)

y_predicted4 = KNN_model.predict(X_test3)


y_test = np.array(y_test, dtype = np.float)
y_train = np.array(y_train, dtype = np.float)
print("Accuracy:  %9f" % metrics.accuracy_score(y_test, y_predicted4))
print("Precision:  %2f" % metrics.precision_score(y_test, y_predicted4, average='macro'))
print("Recall:  %11f" % metrics.recall_score(y_test, y_predicted4, average='macro'))
print("F1:  %15f \n" % metrics.f1_score(y_test, y_predicted4, average='macro'))
   
'''       DTREE MODEL ON NEW DATA ''' 
#         DTREE MODEL ON NEW DATA 

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth = 15, random_state=0)

tree.fit(X_train3 ,y_train)

y_predicted5 = tree.predict(X_test3)
   
   
y_test = np.array(y_test, dtype = np.float)
y_train = np.array(y_train, dtype = np.float)
print("Accuracy:  %9f" % metrics.accuracy_score(y_test, y_predicted5))
print("Precision:  %2f" % metrics.precision_score(y_test, y_predicted5, average='macro'))
print("Recall:  %11f" % metrics.recall_score(y_test, y_predicted5, average='macro'))
print("F1:  %15f \n" % metrics.f1_score(y_test, y_predicted5, average='macro'))
   
   
'''       NN MODEL ON NEW DATA '''  
#         NN MODEL ON NEW DATA 


from sklearn.neural_network import MLPClassifier

  
clf = MLPClassifier(hidden_layer_sizes=150, batch_size = 100, random_state=0, max_iter=300).fit(X_train3, y_train)  
  
    
y_predicted6 = clf.predict(X_test3)
    
y_test = np.array(y_test, dtype = np.float)
y_train = np.array(y_train, dtype = np.float)
print("Accuracy:  %9f" % metrics.accuracy_score(y_test, y_predicted6))
print("Precision:  %2f" % metrics.precision_score(y_test, y_predicted6, average='macro'))
print("Recall:  %11f" % metrics.recall_score(y_test, y_predicted6, average='macro'))
print("F1:  %15f \n" % metrics.f1_score(y_test, y_predicted6, average='macro'))





X_train, X_test, y_train, y_test = model_selection.train_test_split(XX, y, test_size = 0.25, random_state = 0)

'''  START OF CITATION-KNN !!!!!!!!!! '''
#### START OF CITATION-KNN !!!!!!!!!!

model= CitationKNN()
model.fit(X_train, y_train)
y_predicted3 = model.predict(X_test)

y_test3 = np.array(y_test, dtype = np.float)

print("Accuracy:  %9f" % metrics.accuracy_score(y_test3, y_predicted3))
print("Precision:  %2f" % metrics.precision_score(y_test3, y_predicted3, average='macro'))
print("Recall:  %11f" % metrics.recall_score(y_test3, y_predicted3, average='macro'))
print("F1:  %15f \n" % metrics.f1_score(y_test3, y_predicted3, average='macro'))

'''  END OF CITATION-KNN !!!!!!!!!! '''
#### END OF CITATION-KNN !!!!!!!!!!


'''  Multi-Instance-Support-Vector-Machine (Multi-class) !!!!!!!!!! '''
#### Multi-Instance-Support-Vector-Machine (Multi-class) !!!!!!!!!!

print("\nGive again the same Inputs for the misvm algorithm\n")
c = int(input("Please if you want to run the first dataset give 1, otherwise give 2 for the second dataset:   "))
g = int(input("Please give 1 for the NBF generator, give 2 for the ROW generator, give 3 for the k-meansSeg generator:   "))


if c == 1:
    
    ''' the path of the first dataset '''
    #   the path of the first dataset
    path = (r'C:\Users\Thanasis\Desktop\project_advance_topics_in_ML\miml-image-data\original')
   
    yy = np.zeros(1200)
    yy[0:400] = 1
    yy[400:800] = 2
    yy[800:1200] = 3

    
    onlyfiles2 = [ f for f in listdir(path) if isfile(join(path,f)) ]
    onlyfiles = [] 
    XX = []
    
    for n in range(0, (len(onlyfiles2))):
        jpg = '.jpg'
        string = str(n+1)
        onlyfiles.append(string+jpg)
    
    for i in range(1200):
        filepath = path + '\\' + onlyfiles[i]
        # filepath = Path(filepath)
        image = cv2.imread(filepath)
        if g == 3:
            image= cv2.resize(image, (100, 100))
            
        if g == 1:
            bag = generator(image)
        elif g == 2:
            bag = generator2(image)
        elif g == 3:
            bag = generator3(image)
        else:
            print("\nError: You gave wrong Input !!!\n")
            exit()
        
        
        XX.append(bag)
        print(i+1)
    
elif c == 2:
    
    ''' the path of the seconf dataset '''
    #   the path of the second dataset
    path = (r'C:\Users\Thanasis\Desktop\project_advance_topics_in_ML\new_data\dataset2')
    
    yy = np.zeros(766)
    yy[0:300] = 1
    yy[300:514] = 2
    yy[514:766] = 3
    
    onlyfiles2 = [ f for f in listdir(path) if isfile(join(path,f)) ]
    onlyfiles = [] 
    XX = []

    for i in range(766):
        filepath = path + '\\' + onlyfiles2[i]
        # filepath = Path(filepath)
        image = cv2.imread(filepath)
        if g == 3:
            image= cv2.resize(image, (100, 100))
        
        if g == 1:
            bag = generator(image)
        elif g == 2:
            bag = generator2(image)
        elif g == 3:
            bag = generator3(image)
        else:
            print("\nError: You gave wrong Input !!!\n")
            exit()
        
        XX.append(bag)
        print(i+1)
        
else:
    print("\nError: You gave wrong Input !!!\n")
    exit()
 
    


X_train, X_test, y_train, y_test = model_selection.train_test_split(XX, yy, test_size = 0.25, random_state = 0)
y = []
X = []

''' model 1'''
# model 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
y = []
X = []

for i in range(len(y_train)):
    if y_train[i] == 1:
        X.append(X_train[i])
        y.append(1)
    if y_train[i] == 2:
        X.append(X_train[i])
        y.append(-1)
    
y = np.array(y)
y = np.array(y, dtype = np.float)   

classifier1 = misvm.MISVM(kernel='quadratic',  C=70.0)   
classifier1.fit(X, y) 



''' model 2'''
# model 2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

y = []
X = []

for i in range(len(y_train)):
    if y_train[i] == 1:
        X.append(X_train[i])
        y.append(1)
    if y_train[i] == 3:
        X.append(X_train[i])
        y.append(-1)
    
y = np.array(y)
y = np.array(y, dtype = np.float)   


classifier2 = misvm.MISVM(kernel='quadratic',  C=70.0)   
classifier2.fit(X, y) 


''' model 3'''
# model 3!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

y = []
X = []

for i in range(len(y_train)):
    if y_train[i] == 2:
        X.append(X_train[i])
        y.append(1)
    if y_train[i] == 1:
        X.append(X_train[i])
        y.append(-1)
    
y = np.array(y)
y = np.array(y, dtype = np.float)   


classifier3 = misvm.MISVM(kernel='quadratic',  C=70.0)   
classifier3.fit(X, y) 


''' model 4'''
# model 4 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

y = []
X = []

for i in range(len(y_train)):
    if y_train[i] == 2:
        X.append(X_train[i])
        y.append(1)
    if y_train[i] == 3:
        X.append(X_train[i])
        y.append(-1)
    
y = np.array(y)
y = np.array(y, dtype = np.float)   


classifier4 = misvm.MISVM(kernel='quadratic',  C=70.0)   
classifier4.fit(X, y) 


''' model 5'''
# model 5 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

y = []
X = []

for i in range(len(y_train)):
    if y_train[i] == 3:
        X.append(X_train[i])
        y.append(1)
    if y_train[i] == 1:
        X.append(X_train[i])
        y.append(-1)
    
y = np.array(y)
y = np.array(y, dtype = np.float)   


classifier5 = misvm.MISVM(kernel='quadratic',  C=70.0)   
classifier5.fit(X, y) 


''' model 6'''
# model 6 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

y = []
X = []

for i in range(len(y_train)):
    if y_train[i] == 3:
        X.append(X_train[i])
        y.append(1)
    if y_train[i] == 2:
        X.append(X_train[i])
        y.append(-1)
    
y = np.array(y)
y = np.array(y, dtype = np.float)   


classifier6 = misvm.MISVM(kernel='quadratic',  C=70.0)   
classifier6.fit(X, y) 

''' model 1_1'''
# model 1_1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
y = []
X = []

for i in range(len(y_train)):
    if y_train[i] == 1:
        X.append(X_train[i])
        y.append(1)
    else:
        X.append(X_train[i])
        y.append(-1)
    
y = np.array(y)
y = np.array(y, dtype = np.float)   

classifier11 = misvm.MISVM(kernel='quadratic',  C=70.0)   
classifier11.fit(X, y) 

''' model 2_2'''
# model 2_2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
y = []
X = []

for i in range(len(y_train)):
    if y_train[i] == 2:
        X.append(X_train[i])
        y.append(1)
    else:
        X.append(X_train[i])
        y.append(-1)
    
y = np.array(y)
y = np.array(y, dtype = np.float)   

classifier22 = misvm.MISVM(kernel='quadratic',  C=70.0)   
classifier22.fit(X, y) 


''' model 3_3'''
# model 3_3 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
y = []
X = []

for i in range(len(y_train)):
    if y_train[i] == 3:
        X.append(X_train[i])
        y.append(1)
    else:
        X.append(X_train[i])
        y.append(-1)
    
y = np.array(y)
y = np.array(y, dtype = np.float)   

classifier33 = misvm.MISVM(kernel='quadratic',  C=70.0)   
classifier33.fit(X, y) 


y_predicted1 = classifier1.predict(X_test)
y_predicted2 = classifier2.predict(X_test)
y_predicted3 = classifier3.predict(X_test)
y_predicted4 = classifier4.predict(X_test)
y_predicted5 = classifier5.predict(X_test)
y_predicted6 = classifier6.predict(X_test)

y_predicted11 = classifier11.predict(X_test)
y_predicted22 = classifier22.predict(X_test)
y_predicted33 = classifier33.predict(X_test)
y_predicted11 = np.sign(y_predicted11)
y_predicted22 = np.sign(y_predicted22)
y_predicted33 = np.sign(y_predicted33)


y_predicted1 = np.sign(y_predicted1)
y_predicted2 = np.sign(y_predicted2)
y_predicted3 = np.sign(y_predicted3)
y_predicted4 = np.sign(y_predicted4)
y_predicted5 = np.sign(y_predicted5)
y_predicted6 = np.sign(y_predicted6)


'''  misvm to multi-class by using the above models 1v1 and 1vsall '''
#### misvm to multi-class by using the above models 1v1 and 1vsall

y_pred=np.zeros(len(y_test))
for i in range(len(y_test)):
    
    if (y_predicted1[i] == y_predicted2[i] == 1) and ((y_predicted3[i] == y_predicted4[i] == -1) or y_predicted3[i] != y_predicted4[i] ) and ((y_predicted5[i] == y_predicted6[i] == -1) or y_predicted5[i] != y_predicted6[i] ):
        y_pred[i] = 1
    
    elif (y_predicted3[i] == y_predicted4[i] == 1) and ((y_predicted1[i] == y_predicted2[i] == -1) or y_predicted1[i] != y_predicted2[i] ) and ((y_predicted5[i] == y_predicted6[i] == -1) or y_predicted5[i] != y_predicted6[i] ):
        y_pred[i] = 2
        
    elif (y_predicted5[i] == y_predicted6[i] == 1) and ((y_predicted1[i] == y_predicted2[i] == -1) or y_predicted1[i] != y_predicted2[i] ) and ((y_predicted3[i] == y_predicted4[i] == -1) or y_predicted3[i] != y_predicted4[i] ):                 
        y_pred[i] = 3
        
    
    
    elif ((y_predicted1[i] == 1 or y_predicted2[i] == 1) and y_predicted3[i] == -1 and y_predicted4[i] == -1 and y_predicted5[i] == -1 and y_predicted6[i] == -1):
        y_pred[i] =1
        
    elif ((y_predicted3[i] == 1 or y_predicted4[i] == 1) and y_predicted1[i] == -1 and y_predicted2[i] == -1 and y_predicted5[i] == -1 and y_predicted6[i] == -1):
        y_pred[i] =2
        
    elif ((y_predicted5[i] == 1 or y_predicted6[i] == 1) and y_predicted3[i] == -1 and y_predicted4[i] == -1 and y_predicted1[i] == -1 and y_predicted2[i] == -1):
        y_pred[i] =3
        
        
        
    elif (y_predicted1[i] == y_predicted2[i] == 1) and (y_predicted3[i] == y_predicted4[i] == 1) and ((y_predicted5[i] == y_predicted6[i] == -1) or y_predicted5[i] != y_predicted6[i] ):
        
        if y_predicted11[i]==1 and y_predicted22[i]==-1:
            y_pred[i] = 1
        elif y_predicted11[i]==-1 and y_predicted22[i]==1:
            y_pred[i] = 2
            
    elif (y_predicted1[i] == y_predicted2[i] == 1) and (y_predicted5[i] == y_predicted6[i] == 1) and ((y_predicted3[i] == y_predicted4[i] == -1) or y_predicted3[i] != y_predicted4[i] ):
        
        if y_predicted11[i]==1 and y_predicted33[i]==-1:
            y_pred[i] = 1
        elif y_predicted11[i]==-1 and y_predicted33[i]==1:
            y_pred[i] = 3
            
    elif (y_predicted3[i] == y_predicted4[i] == 1) and (y_predicted5[i] == y_predicted6[i] == 1) and ((y_predicted1[i] == y_predicted2[i] == -1) or y_predicted1[i] != y_predicted2[i] ):
        
        if y_predicted22[i]==1 and y_predicted33[i]==-1:
            y_pred[i] = 2
        elif y_predicted22[i]==-1 and y_predicted33[i]==1:
            y_pred[i] = 3
            
    
    elif (y_predicted1[i] != y_predicted2[i]) and (y_predicted3[i] != y_predicted4[i] ) and ((y_predicted5[i] == y_predicted6[i] == -1) ):
        
        if y_predicted11[i]==1 and y_predicted22[i]==-1:
            y_pred[i] = 1
        elif y_predicted11[i]==-1 and y_predicted22[i]==1:
            y_pred[i] = 2
            
    elif (y_predicted1[i] != y_predicted2[i]) and (y_predicted5[i] != y_predicted6[i]) and ((y_predicted3[i] == y_predicted4[i] == -1)):
        
        if y_predicted11[i]==1 and y_predicted33[i]==-1:
            y_pred[i] = 1
        elif y_predicted11[i]==-1 and y_predicted33[i]==1:
            y_pred[i] = 3
            
    elif (y_predicted3[i] != y_predicted4[i]) and (y_predicted5[i] != y_predicted6[i]) and ((y_predicted1[i] != y_predicted2[i] == -1) ):
        
        if y_predicted22[i]==1 and y_predicted33[i]==-1:
            y_pred[i] = 2
        elif y_predicted22[i]==-1 and y_predicted33[i]==1:
            y_pred[i] = 3
         
     
    else:
        
        pass
        if y_predicted11[i]==1 and y_predicted22[i]==-1 and y_predicted33[i]==-1 :
            y_pred[i] = 1
          
        if y_predicted11[i]==-1 and y_predicted22[i]==1 and y_predicted33[i]==-1 :
            y_pred[i] = 2
            
        if y_predicted11[i]==-1 and y_predicted22[i]==-1 and y_predicted33[i]==1 :
            y_pred[i] = 3
            
    




print("Accuracy:  %9f" % metrics.accuracy_score(y_test, y_pred))
print("Precision:  %2f" % metrics.precision_score(y_test, y_pred, average='macro'))
print("Recall:  %11f" % metrics.recall_score(y_test, y_pred, average='macro'))
print("F1:  %15f \n" % metrics.f1_score(y_test, y_pred, average='macro'))


count=0
for i in range(len(y_pred)):
    if y_pred[i] == 0:
        count = count+1
        
        

'''  Multi-Instance-Support-Vector-Machine only in one dataset !!!!!!!!!! '''
#### Multi-Instance-Support-Vector-Machine only in one dataset !!!!!!!!!!


# c = int(input("Please if you want to run the first dataset give 1, otherwise give 2 for the second dataset:   "))
g = int(input("Please give 1 for the NBF generator, give 2 for the ROW generator, give 3 for the k-meansSeg generator:   "))


''' the path of the first dataset '''
#   the path of the first dataset
path = (r'C:\Users\Thanasis\Desktop\project_advance_topics_in_ML\new_data\dataset2') 

yy = np.zeros(1123)
yy[0:300] = -1
yy[300:514] = -1
yy[514:766] = 1
yy[766:1123] = 1

    
onlyfiles2 = [ f for f in listdir(path) if isfile(join(path,f)) ]
onlyfiles = [] 
XX = []

for i in range(1123):
    filepath = path + '\\' + onlyfiles2[i]
    # filepath = Path(filepath)
    image = cv2.imread(filepath)
    if g == 3:
        image= cv2.resize(image, (100, 100))
        
    if g == 1:
        bag = generator(image)
    elif g == 2:
        bag = generator2(image)
    elif g == 3:
        bag = generator3(image)
    else:
        print("\nError: You gave wrong Input !!!\n")
        exit()
        
    XX.append(bag)
    print(i+1)

    
X_train, X_test, y_train, y_test = model_selection.train_test_split(XX, yy, test_size = 0.25, random_state = 0)


classifier = misvm.MISVM(kernel='quadratic',  C=70.0)   
classifier.fit(X_train, y_train)   
y_predicted = classifier.predict(X_test)

y_predicted = np.sign(y_predicted)


print("Accuracy:  %9f" % metrics.accuracy_score(y_test, y_predicted))
print("Precision:  %2f" % metrics.precision_score(y_test, y_predicted, average='macro'))
print("Recall:  %11f" % metrics.recall_score(y_test, y_predicted, average='macro'))
print("F1:  %15f \n" % metrics.f1_score(y_test, y_predicted, average='macro'))






