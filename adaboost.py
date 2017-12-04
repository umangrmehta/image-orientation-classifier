#!/usr/bin/python

import sys
import math
import operator
from Queue import PriorityQueue
import numpy as np
from knn import *
import numpy as np

def adaboostTrain(trainFile, modelFile):
    modelappend = open(modelFile, "w+")
    modelFileKnn = modelFile+".ada"
    knnTrain(trainFile, modelFileKnn)
    model = open(modelFileKnn, "r")
    lineNumber = 0
    kValue = 7
    numLinesTrain = sum(1 for line in open(modelFileKnn))
    trainVector = np.zeros((numLinesTrain, 192), dtype=np.int_)
    trainOrient = np.zeros((numLinesTrain, 1), dtype=np.int_)
    testVector = np.zeros((numLinesTrain, 192), dtype=np.int_)
    testOrient = np.zeros((numLinesTrain, 1), dtype=np.int_)
    for row in model:
        rowList = row.split('|')
        intmOrient = rowList[0]
        intmVector = rowList[1].split(' ')
        vector = [int(i) for i in intmVector]
        trainOrient[lineNumber] = intmOrient
        trainVector[lineNumber] = np.array(vector)
        lineNumber += 1
    model.close()
    alphalist = []
    wtlist = np.full((numLinesTrain, 1),(1.0 / numLinesTrain), dtype=np.float_)
    trainData = open(trainFile, "r")
    lineNumber = 0
    for row in trainData:
        rowList = row[:-1].split(' ', 2)
        intmOrient = rowList[1]
        intmVector = rowList[2].split(' ')
        vector = [int(i) for i in intmVector]
        testOrient[lineNumber] = intmOrient
        testVector[lineNumber] = np.array(vector)
        lineNumber += 1
    trainData.close()
    for element in [0,90,180,270]:
        wc = 0
        wnc = 0
        correctnesslist = np.zeros((numLinesTrain, 1), dtype=np.int_)
        lineNumber = 0
        for item in range(0,len(testOrient),1):
            print "--------------Line" +str(lineNumber)+ "-----------------"
            vectorlist = testVector[item]
            orientval = testOrient[item]
            pred = knnTest(vectorlist, trainOrient, trainVector, kValue)
            if pred == element and orientval == element:
                correctnesslist[lineNumber] = True
                wc += 1
            elif pred != element and orientval != element:
                correctnesslist[lineNumber] = True
                wc += 1
            else:
                correctnesslist[lineNumber] = False
                wnc += 1
            print correctnesslist[lineNumber]
            lineNumber+=1
        alpha = (wc+1)/float(wnc+1)
        alphalist.append(alpha)
        for el in range(len(wtlist)):
            if correctnesslist[el] == True:
                wtlist[el] = math.exp(-1*alpha)
            else:
                wtlist[el] = math.exp(alpha)
        z_t = np.sum(wtlist)
        wtlist /= z_t
        e_t = wc/float(wc+wnc)*100
        print e_t
        print wc
        print alphalist
        trainData.close()
    modelappend.write("Completed..")

    # for sth in range(0,len(vectorlist)-1):
    #     wtlist.append(wt)
    # modelAppend.write(wtlist)
    # modelAppend.close()

def adaboostTest(trainFile, modelFile):
    pass
