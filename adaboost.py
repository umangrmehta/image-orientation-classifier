#!/usr/bin/python

import sys
import math
import operator
from Queue import PriorityQueue
import numpy as np

def adaboostTrain(trainFile, modelFile):
    vectorlist = []
    orientlist = [] #membership function
    wt = 1 #start with 1
    wtlist = [] #weight list for corresponding vector
    trainData = open(trainFile, "r")
    modelAppend = open(modelFile, "w+")
    for line in trainData:
        row = line[:-1].split(' ', 2)
        vectorlist.append(row[2:])
        orientlist.append(row[1])
        print vectorlist
    for sth in range(0,len(vectorlist)-1):
        wtlist.append(wt)
    modelAppend.write(wtlist)
    modelAppend.close()

def adaboostTest(trainFile, modelFile):
    pass
