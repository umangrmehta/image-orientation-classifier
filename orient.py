#!/usr/bin/python

import sys
import math
import operator
from Queue import PriorityQueue
import numpy as np

from adaboost import *
from knn import *

def knnTestPreprocessor():
    lineNumber = 0
    model = open(modelFile, "r")
    for row in model:
        rowList = row.split('|')
        intmOrient = rowList[0]
        intmVector = rowList[1].split(' ')
        vector = [int(i) for i in intmVector]
        trainOrient[lineNumber] = intmOrient
        trainVector[lineNumber] = np.array(vector)
        lineNumber += 1
    model.close()
    lineNumber = 0
    testData = open(switchFile, "r")
    for line in testData:
        testList = line.split(' ')
        testFile[lineNumber] = testList[0]
        testList = [int(i) for i in testList[1:]]
        testOrient[lineNumber] = testList[0]
        testVector[lineNumber] = np.array(testList[1:])
        lineNumber += 1
    testData.close()

switch = sys.argv[1]
switchFile = sys.argv[2]
modelFile = sys.argv[3]
model = sys.argv[4]
output = open("output.txt", "r")

if model.lower() == "nearest":
    if switch.lower() == "train":
        knnTrain(switchFile, modelFile)
    if switch.lower() == "test":
        accuracy = 0
        numLinesTrain = sum(1 for line in open(modelFile))
        numLinesTest = sum(1 for line in open(switchFile))
        trainVector = np.zeros((numLinesTrain, 192), dtype=np.int_)
        trainOrient = np.zeros((numLinesTrain, 1), dtype=np.int_)
        testVector = np.zeros((numLinesTest, 192), dtype=np.int_)
        testOrient = np.zeros((numLinesTest, 1), dtype=np.int_)
        testFile = np.empty(numLinesTest, dtype='S256')
        knnTestPreprocessor()
        for row in range(0, len(testOrient), 1):
            knn = {0: 0, 90: 0, 180: 0, 270: 0}
            knnDist = {0: 0, 90: 0, 180: 0, 270: 0}
            # print "-------------LINE " + str(row) + "-------------"
            predictOrient = knnTest(testVector[row], trainOrient, trainVector, 45, knn, knnDist)
            output.write("%s %s\n" % (str(testFile[row]), str(predictOrient)))
            accuracy += (1 if predictOrient == int(testOrient[row]) else 0)
        print "K-Nearest Neighbours Accuracy: " + str(100.0 * accuracy / row)
elif model.lower() == "adaboost":
    if switch.lower() == "train":
        adaboostTrain(switchFile, modelFile)
    if switch.lower() == "test":
        adaboostTest(switchFile, modelFile)

output.close()