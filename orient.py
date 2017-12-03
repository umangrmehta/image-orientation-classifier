#!/usr/bin/python

import sys
import math
import operator
from Queue import PriorityQueue

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
    distQueue = PriorityQueue()
    testData = open(testFile, "r")
    model = open(modelFile, "r")
    print "Begin test processing.."
    n = 0
    accuracy = 0
    for line in testData:
        knn = {'0': 0, '90': 0, '180': 0, '270': 0}
        n += 1
        print "--------------------------------------------------------------------------------------------------------------" + str(n)
        testList = line.split(' ')
        testOrient = testList[1]
        testVector = testList[2:]
        for row in model:
            sumEucDist = 0
            rowList = row.split('|')
            vector = rowList[1].split(' ')
            orient = rowList[0]
            for i in range(0, 192, 1):
                sumEucDist = sumEucDist + math.pow(int(testVector[i]) - int(vector[i]), 2)
            eucDist = math.sqrt(sumEucDist)
            distQueue.put((eucDist, orient))
        k=35
        for i in range(0, k, 1):
            knnOrient = distQueue.get()
            knn[knnOrient[1]] += 1
            predictOrient = max(knn, key=knn.get)
        accuracy += (1 if int(predictOrient) == int(testOrient) else 0)
        if predictOrient != testOrient:
            print predictOrient
            print testOrient
            print knn
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
