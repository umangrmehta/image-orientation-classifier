#!/usr/bin/python

import random
import numpy as np

opPositionVector = ['0', '90', '180', '270']


def sigmoid(x, derivative=False):
	return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def train(trainFile, modelFile):
	trainData = open(trainFile, "r")
	modelAppend = open(modelFile, "w")

	trainDataLength = sum(1 for l in trainData)
	trainData = open(trainFile, "r")

	trainIPVectors = np.zeros((trainDataLength,  192), dtype=np.int_)
	trainOPVectors = np.zeros((trainDataLength,  4), dtype=np.int8)
	for lineNo, line in enumerate(trainData):
		row = line[:-1].split(' ', 2)
		trainIPVectors[lineNo] = np.array(row[2].split(' '))
		trainOPVectors[lineNo, opPositionVector.index(row[1])] = 1

	ipToHidden = np.random.rand(192, 20)
	hiddenToOP = np.random.rand(20, 4)

	for i in range(0, 1000):
		vectorCheck = np.zeros(trainDataLength, dtype=np.bool_)
		while not np.all(vectorCheck):
			vectorIDX = random.randint(0, trainDataLength - 1)
			if vectorCheck[vectorIDX]:
				continue
			ipVector = trainIPVectors[vectorIDX]
			# Feed Forward
			hidden = np.matmul(ipVector, ipToHidden)
			hiddenOP = sigmoid(hidden)

			op = np.matmul(hiddenOP, hiddenToOP)
			finalOP = sigmoid(op)

			# Back-Propagation
			opError = sigmoid(np.asarray(op), derivative=True) * np.asarray(trainOPVectors[vectorIDX] - finalOP)
			hiddenError = sigmoid(np.asarray(hidden), derivative=True) * np.asarray(np.matmul(opError, hiddenToOP.transpose()))

			hiddenToOP = hiddenToOP + np.multiply(0.1, np.matmul(np.asmatrix(hiddenOP).transpose(), np.asmatrix(opError)))
			ipToHidden = ipToHidden + np.multiply(0.1, np.matmul(np.asmatrix(ipVector).transpose(), np.asmatrix(hiddenError)))

			vectorCheck[vectorIDX] = True

	print np.asmatrix(ipToHidden)
	print np.asmatrix(hiddenToOP)


def test(testFile, modelFile):
	pass
