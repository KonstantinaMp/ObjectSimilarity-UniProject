#ΜΠΟΥΡΟΓΙΑΝΝΗ ΚΩΝΣΤΑΝΤΙΝΑ - ΑΜ: 2775 - cs02775

import sys 
import os
import csv
import random
import math
import copy
import itertools
import time
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import more_itertools as mit  #pip install more_itertools
import numpy as np
import pandas as pd

from universalHashFunctions import *
from randomPermutation import *
from itertools import islice
from itertools import zip_longest

inFile100 = 'ratings_100users.csv'
inFile610 = 'ratings.csv'

#(1a) Pre-processing
#==================================================================================================
def readFile(file,usersNo):
    
    movieNo = 0

    userList = {}
    movieMap = {}
    movieList = {} 
    
    for user in range(1, usersNo+1):    
        userList[user] = list()

    inFile = open(file,'r')
    inFile.readline()
    for line in inFile:
        userId, movieId = line.split(',')[:-2]
        userId = int(userId)
        movieId = int(movieId)
        userList[int(userId)].append(int(movieId))
        
        if movieId not in movieMap.keys():
            movieNo +=1
            movieMap[movieId] = movieNo 
        if movieId not in movieList.keys():
            movieList[movieId] = [userId]
        else:
            movieList[movieId].append(userId)
    inFile.close()
    
    return [userList,movieMap,movieList]

#(1b) Compute Jaccard-Similarity
#==================================================================================================
def jaccardSimilarity(movieId1,movieId2):  
    
    s1 = set( movieList[movieId1] )
    s2 = set( movieList[movieId2] ) 
    JacSim = ( len(s1.intersection(s2)) / len(s1.union(s2)) )
    
    return JacSim

#(1c) Min-Hash Signatures
#==================================================================================================
def minHash(n):
    
    sigTemp = {}
    sig = dict.fromkeys(movieMap)
    
    for key in sig.keys():
        sig[key] = []

    for j in range(0,n):
        sigTemp = dict.fromkeys(movieMap ,math.inf)
        returnLists = create_random_permutation(K=usersNo)
        hashDict = dict(returnLists[0])
        randomPermutation = create_random_permutation(K=usersNo)[1]
        for i in randomPermutation:
            for col in userList[i]:
                if hashDict[i] < sigTemp[col] :
                    sigTemp[col] = hashDict[i]
                sigTemp[col] = randomPermutation.index(i)
        
        for i in sig.keys():
           sig[i].append(sigTemp[i])
    
    return sig

    

#(1d) Signature Similarity
#==================================================================================================
def signatureSimilarity(movieId1,movieId2,sig,n):
    
    counter = 0
    
    hashValues1 = sig[movieId1][:n]
    hashValues2 = sig[movieId2][:n]
    
    for i in range(0,len(hashValues1)):
        if hashValues1[i] == hashValues2[i]:
            counter+=1
    
    SigSim = counter/n
    
    return SigSim

#(1e) Locality-Sensitive Hashing : Candidate pairs of
#==================================================================================================
def LSH(sig,s,n,r,b):

    bands = []
    candidatePairs = []
    sigTemp = {}
    
    h = create_random_hash_function(p=2**33-355, m=2**32-1)
    
    for i in range(0,b):
        bands.append(dict.fromkeys(movies)) #movieList
    for band in bands:
        for key in band.keys():
            band[key] = []
        
    for i in movies: #movieList
        sigTemp[i] = sig[i][:n]
    
    for col in sigTemp:
        i=0
        for x in mit.divide(b, sigTemp[col]):
            bands[i][col] = bands[i][col] + list(x)
            i+=1
    
    for band in bands:
        bucketTemp = {}
        for movie in band:
            vector = ''
            for v in band[movie]:
                if len(str(v)) == 1:
                    vector=vector+'0'+str(v)
                else:   
                    vector = vector+str(v)
            bucket = h(int(vector))
            if bucket in bucketTemp.keys():
                bucketTemp[bucket].append(movie)
            else:
                bucketTemp[bucket] = [movie]
        
        for i in bucketTemp.keys():
            if len(bucketTemp[i])>=2:
                for pair in itertools.combinations(bucketTemp[i],2):
                    if pair not in candidatePairs:
                        candidatePairs.append(pair)

    return candidatePairs
     

# MAIN
#==================================================================================================
def main(argv):

    global userList 
    global movieMap
    global movieList 
    global usersNo
    
    global N
    global movies
    

    userList = {}
    movieMap = {}
    movieList = {}
    movies = {}
    SIG = {}
    jacS = {}        #dicionary movies pair:jaccard similarity
    sigS ={}         #dicionary movies pair:signature similarity
    candidatePairs = []  

    allN = [5,10,15,20,25,30,35,40]
    scores = {5:[0,0,0,0],10:[0,0,0,0],15:[0,0,0,0],20:[0,0,0,0],25:[0,0,0,0],30:[0,0,0,0],35:[0,0,0,0],40:[0,0,0,0]}   #dicitionary n:[num of fp,num of fn,num of tp, num of tn]
    metrics = [[],[],[]]   #each sublist(len=8) stores the precision,recall and f1 value for all n in range(5,40)
    
    lshArgs = [(2,20),(4,10),(5,8),(8,5),(10,4),(20,2)]
    lshScores = {(2,20):[0,0,0,0],(4,10):[0,0,0,0],(5,8):[0,0,0,0],(8,5):[0,0,0,0],(10,4):[0,0,0,0],(20,2):[0,0,0,0]}   #dicitionary n:[num of fp,num of tp,num of tn,num of fn]
    lshMetrics = [[],[],[]]   #each sublist(len=6) stores the precision,recall and f1 value of each lsh (r,b) combination
    

    if sys.argv[1] == '100': 
        inFile = inFile100
        usersNo = 100
        N = 20
    elif sys.argv[1] == '610':
        inFile = inFile610
        usersNo = 610
        N = 100

    
    data = readFile(inFile,usersNo)
    userList = data[0].copy()
    movieMap = data[1].copy()
    movieList = data[2].copy()   

    for key in sorted(movieList.keys())[0:N]:
        movies[key] = movieList[key]
 
    for pair in itertools.combinations(movies,2):
        jacS[pair] = jaccardSimilarity(pair[0],pair[1])

    SIG = pd.DataFrame(minHash(30).copy())
    
    for n in range(5,45,5):
        sigS[n] = dict.fromkeys(jacS)
        for pair in jacS.keys():
            sigS[n][pair]=signatureSimilarity(pair[0],pair[1],SIG,n)
            if sigS[n][pair] >= 0.25 and jacS[pair] < 0.25:     #false pos
                scores[n][0]+=1    
            elif sigS[n][pair] < 0.25 and jacS[pair] >= 0.25:   #false neg
                scores[n][1]+=1
            elif sigS[n][pair] >= 0.25 and jacS[pair] >= 0.25:    #true pos
                scores[n][2]+=1
            else:                                               #true neg
                scores[n][3]+=1

    for n in range(5,45,5):
        precision = scores[n][2] / (scores[n][2] + scores[n][0])
        metrics[0].append(precision)
        recall = scores[n][2] / (scores[n][2] + scores[n][1])
        metrics[1].append(recall)
        f1 = 2 * recall * precision / (recall + precision)
        metrics[2].append(f1)
    

    for inp in lshArgs:
        r = inp[0]
        b = inp[1]
        candidatePairs = LSH(SIG,0.25,40,r,b)
        for pair in candidatePairs:
            if jacS[pair] < 0.25:
                lshScores[(r,b)][0] += 1  #false pos
            elif jacS[pair] >= 0.25:
                lshScores[(r,b)][1] += 1  #true pos
        for pair in jacS.keys():
            if pair not in candidatePairs and jacS[pair] < 0.25:
                lshScores[(r,b)][2] += 1  #true neg
            elif pair not in candidatePairs and jacS[pair] >= 0.25:
                lshScores[(r,b)][3] += 1  #false neg


        precision = scores[40][1] / (scores[40][1] + scores[40][0]) 
        lshMetrics[0].append(precision)
        recall = scores[40][1] / (scores[40][1] + scores[40][3]) 
        lshMetrics[1].append(recall)
        f1 = 2 * recall * precision / (recall + precision) 
        lshMetrics[2].append(f1)

if __name__ == "__main__":
    main(sys.argv)
    
    