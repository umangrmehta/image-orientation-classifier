#!/usr/bin/python

import sys
import math
import operator
from Queue import PriorityQueue
import numpy as np

from adaboost import *
from knn import *

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
    if switch.lower() == "test":
        adaboostTest(switchFile, modelFile)

