#!/usr/bin/python

import sys
import math
import operator
from Queue import PriorityQueue
import numpy as np

def knnTrain(trainFile, modelFile):
    print "Learning Model.."
    trainData = open(trainFile, "r")
    modelAppend = open(modelFile, "w")
    for line in trainData:
        row = line[:-1].split(' ', 2)
        modelAppend.write("%s|%s\n" % (row[1], row[2]))
    trainData.close()
    modelAppend.close()
    print "Learning complete.."

def knnTest(testFile, modelFile):
    lineNumber = 0
    testData = open(testFile, "r")
    num_lines = sum(1 for line in open(modelFile))
    trainVector = np.zeros((num_lines+1, 192), dtype=np.int_)
    trainOrient = np.zeros((num_lines+1, 1), dtype=np.int_)
    model = open(modelFile, "r")
    for row in model:
        lineNumber += 1
        rowList = row.split('|')
        intmOrient = rowList[0]
        intmVector = rowList[1].split(' ')
        vector = [int(i) for i in intmVector]
        trainOrient[lineNumber] = intmOrient
        trainVector[lineNumber] = np.array(vector)
    model.close()
    print "Begin test processing.."
    n = 0
    accuracy = 0
    for line in testData:
        distQueue = PriorityQueue()
        knn = {0: 0, 90: 0, 180: 0, 270: 0}
        knnDist = {0: 0, 90: 0, 180: 0, 270: 0}
        n += 1
        print "--------------------------------------------------------------------------------------------------------------" + str(n)
        testList = line.split(' ')
        testList = [int(i) for i in testList[1:]]
        testOrient = testList[0]
        testVector = np.array(testList[1:])
        model = open(modelFile, "r")
        for row in range(0,len(trainOrient),1):
            vector = trainVector[row]
            orient = int(trainOrient[row])
            eucDist = math.sqrt(np.sum(np.power((vector - testVector), 2)))
            distQueue.put((eucDist, orient))
        model.close()
        k=35
        for i in range(0, k, 1):
            knnScore = distQueue.get()
            knn[knnScore[1]] += 1
            knnDist[knnScore[1]] += knnScore[0]
            predictOrient = max(knn, key=knn.get)
        accuracy += (1 if predictOrient == testOrient else 0)
        print knn
        print knnDist
        if predictOrient != testOrient:
            print predictOrient
            print testOrient
            pass
    print "Accuracy: " + str(100.0 * accuracy / n)

switch = sys.argv[1]
switchFile = sys.argv[2]
modelFile = sys.argv[3]
model = sys.argv[4]

if model.lower() == "nearest":
    if switch.lower() == "train":
        knnTrain(switchFile, modelFile)
    if switch.lower() == "test":
        knnTest(switchFile, modelFile)
    elif model.lower() == "adaboost":
        if switch.lower() == "train":
            adaboostTrain(switchFile, modelFile)
#if switch.lower() == "test":
#adaboostTest(switchFile, modelFile)

# def adaboostTrain(trainFile, modelFile):
#     vectorlist = []
#     orientlist = [] #membership function
#     wt = 1 #start with 1
#     wtlist = [] #weight list for corresponding vector
#     trainData = open(trainFile, "r")
#     modelAppend = open(modelFile, "w+")
#     for line in trainData:
#         row = line[:-1].split(' ', 2)
#         vectorlist.append(row[2:])
#         orientlist.append(row[1])
#         print vectorlist
#     for sth in range(0,len(vectorlist)-1):
#         wtlist.append(wt)
#     modelAppend.write(wtlist)
#     modelAppend.close()
