#!/usr/bin/python

import sys
import math
import operator
from Queue import PriorityQueue

p=2
def knnTrain(trainFile, modelFile):
    trainData = open(trainFile, "r")
    modelAppend = open(modelFile, "w+")
    for line in trainData:
        row = line[:-1].split(' ', 2)
        modelAppend.write("%s|%s\n" % (row[1], row[2]))
    trainData.close()
    modelAppend.close()

def knnTest(testFile, modelFile):
    distQueue = PriorityQueue()
    testData = open(testFile, "r")
    model = open(modelFile, "r")
    n = 0
    accuracy = 0
    for line in testData:
        knn = {'0': 0, '90': 0, '180': 0, '270': 0}
        n += 1
        testList = line.split(' ')
        testOrient = testList[1]
        testVector = testList[2:]
        for row in model:
            sumEucDist = 0
            rowList = row.split('|')
            vector = rowList[1].split(' ')
            orient = rowList[0]
            for i in range(0, 191, 1):
                sumEucDist = sumEucDist + math.pow(abs(int(testVector[i]) - int(vector[i])), p)
            eucDist = math.pow(sumEucDist,round(1/float(p),1))
            distQueue.put((eucDist, orient))
        k=47
        for i in range(0, k, 1):
            knnOrient = distQueue.get()
            knn[knnOrient[1]] += 1/ math.pow(knnOrient[0], 2)
        index, value = max(enumerate(knn), key=operator.itemgetter(1))
        predictOrient = max(knn, key=knn.get)
        accuracy += (1 if int(predictOrient) == int (testOrient) else 0)
        if predictOrient != testOrient:
            print predictOrient
            print testOrient
            print knn
            print "--------------------------------------------------------------------------------------------------------------" + str(n)
    print "Accuracy: " + str(100.0 * accuracy / n)

switch = sys.argv[1]
switchFile = sys.argv[2]
model = sys.argv[3]
modelFile = sys.argv[4]

if model.lower() == "nearest":
    if switch.lower() == "train":
        knnTrain(switchFile, modelFile)
    if switch.lower() == "test":
        knnTest(switchFile, modelFile)
