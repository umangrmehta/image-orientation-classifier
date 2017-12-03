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
        row = line[:-1].split(' ', 1)
        modelAppend.write("%s\n" % row[1])
    trainData.close()
    modelAppend.close()
    print "Learning complete.."

def knnTest(testFile, modelFile):
    distQueue = PriorityQueue()
    testData = open(testFile, "r")
    print "Begin test processing.."
    n = 0
    accuracy = 0
    for line in testData:
        knn = {0: 0, 90: 0, 180: 0, 270: 0}
        knnDist = {0: 0, 90: 0, 180: 0, 270: 0}
        n += 1
        print "--------------------------------------------------------------------------------------------------------------" + str(n)
        testList = line.split(' ')
        testList = [int(i) for i in testList[1:]]
        testOrient = testList[0]
        testVector = np.array(testList[1:])
        model = open(modelFile, "r")
        for row in model:
            vector = row.split(' ')
            vector = [int(i) for i in vector]
            trainVector = np.array(vector[1:])
            trainOrient = int(vector[0])
            eucDist = math.sqrt(sum(np.power((trainVector - testVector), 2)))
            distQueue.put((eucDist, trainOrient))
        model.close()
        k=37
        for i in range(0, k, 1):
            knnScore = distQueue.get()
            knn[knnScore[1]] += 1
            knnDist[knnScore[1]] += knnScore[0]
            predictOrient = max(knn, key=knn.get)
        accuracy += (1 if int(predictOrient) == int(testOrient) else 0)
        print knn
        print knnDist
        print "Accuracy: " + str(100.0 * accuracy / n)
        if predictOrient != testOrient:
            print predictOrient
            print testOrient


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
