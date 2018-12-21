# ----------------------------------------------------------------------
# MIT License
# Copyright (c) 2018 Lee Min Hua. All rights reserved.
# Author: Lee Min Hua
# E-mail: mhlee2907@gmail.com

# This code is originally from Jin-Man Park and Jong-Hwan Kim's Github repository:
# (https://github.com/chickenbestlover/Online-Recurrent-Extreme-Learning-Machine)
# And modified to run Online Recurrent Extreme Learning Machine with Adaptive Forgetting Factor and Genetic Algorithm
# (ORELM-AFF-GA)
# ----------------------------------------------------------------------

"""
Implementation of Online Recurrent Extreme Learning Machine (OR-ELM) with Adaptive Forgetting Factor (AFF)
"""

import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from FOS_ELM import FOSELM
import math

def orthogonalization(Arr):
    [Q, S, _] = np.linalg.svd(Arr)
    tol = max(Arr.shape) * np.spacing(max(S))
    r = np.sum(S > tol)
    Q = Q[:, :r]

def linear_recurrent(features, inputW, hiddenW, hiddenA, bias):
    (numSamples, numInputs) = features.shape
    (numHiddenNeuron, numInputs) = inputW.shape
    V = np.dot(features, np.transpose(inputW)) + np.dot(hiddenA, hiddenW)
    for i in range(numHiddenNeuron):
        V[:, i] += bias[0, i]

    return V

def sigmoidActFunc(V):
    H = 1 / (1 + np.exp(-V))
    return H


class ORELM_AFF(object):
    def __init__(self, inputs, outputs, numHiddenNeurons, activationFunction, min_t, max_t, LN=True, AE=True, ORTH=True,
                 inputWeightForgettingFactor=0.999,
                 outputWeightForgettingFactor=0.999,
                 hiddenWeightForgettingFactor=0.999):

        self.min_t = min_t
        self.max_t = max_t
        self.activationFunction = activationFunction
        self.inputs = inputs
        self.outputs = outputs
        self.numHiddenNeurons = numHiddenNeurons

        # input to hidden weights
        self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
        # hidden layer to hidden layer weights
        self.hiddenWeights = np.random.random((self.numHiddenNeurons, self.numHiddenNeurons))
        # initial hidden layer activation
        self.initial_H = np.random.random((1, self.numHiddenNeurons)) * 2 - 1
        self.H = self.initial_H

        self.LN = LN
        self.AE = AE
        self.ORTH = ORTH

        # bias of hidden units
        self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1

        self.forgettingFactor = outputWeightForgettingFactor

        self.FFmin = 0.9
        self.FFmax = 0.999

        self.trace = 0
        self.thresReset = 0.001

        if self.AE:
            self.inputAE = FOSELM(inputs=inputs,
                                  outputs=inputs,
                                  numHiddenNeurons=numHiddenNeurons,
                                  activationFunction=activationFunction,
                                  LN=LN,
                                  forgettingFactor=inputWeightForgettingFactor,
                                  ORTH=ORTH
                                  )

            self.hiddenAE = FOSELM(inputs=numHiddenNeurons,
                                   outputs=numHiddenNeurons,
                                   numHiddenNeurons=numHiddenNeurons,
                                   activationFunction=activationFunction,
                                   LN=LN,
                                   ORTH=ORTH
                                   )

    def layerNormalization(self, H, scaleFactor=1, biasFactor=0):

        H_normalized = (H - H.mean()) / (np.sqrt(H.var() + 0.000001))
        H_normalized = scaleFactor * H_normalized + biasFactor

        return H_normalized

    def __calculateInputWeightsUsingAE(self, features):
        self.inputAE.train(features=features, targets=features)
        return self.inputAE.beta

    def __calculateHiddenWeightsUsingAE(self, features):
        self.hiddenAE.train(features=features, targets=features)
        return self.hiddenAE.beta

    def calculateHiddenLayerActivation(self, features):
        """
        Calculate activation level of the hidden layer
        :param features feature matrix with dimension (numSamples, numInputs)
        :return: activation level (numSamples, numHiddenNeurons)
        """
        if self.activationFunction is "sig":

            if self.AE:
                self.inputWeights = self.__calculateInputWeightsUsingAE(features)

                self.hiddenWeights = self.__calculateHiddenWeightsUsingAE(self.H)

            V = linear_recurrent(features=features,
                                 inputW=self.inputWeights,
                                 hiddenW=self.hiddenWeights,
                                 hiddenA=self.H,
                                 bias=self.bias)
            if self.LN:
                V = self.layerNormalization(V)
            self.H = sigmoidActFunc(V)

        else:
            print " Unknown activation function type"
            raise NotImplementedError
        return self.H

    def initializePhase(self, lamb=0.0001):
        """
        Step 1: Initialization phase
        :param features feature matrix with dimension (numSamples, numInputs)
        :param targets target matrix with dimension (numSamples, numOutputs)
        """

        if self.activationFunction is "sig":
            self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1
        else:
            print " Unknown activation function type"
            raise NotImplementedError

        self.M = inv(lamb * np.eye(self.numHiddenNeurons))
        self.beta = np.zeros([self.numHiddenNeurons, self.outputs])

        # randomly initialize the input->hidden connections
        self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
        self.inputWeights = self.inputWeights * 2 - 1

        if self.AE:
            self.inputAE.initializePhase(lamb=0.00001)
            self.hiddenAE.initializePhase(lamb=0.00001)
        else:
            # randomly initialize the input->hidden connections
            self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
            self.inputWeights = self.inputWeights * 2 - 1

            if self.ORTH:
                if self.numHiddenNeurons > self.inputs:
                    self.inputWeights = orthogonalization(self.inputWeights)
                else:
                    self.inputWeights = orthogonalization(self.inputWeights.transpose())
                    self.inputWeights = self.inputWeights.transpose()

            # hidden layer to hidden layer wieghts
            self.hiddenWeights = np.random.random((self.numHiddenNeurons, self.numHiddenNeurons))
            self.hiddenWeights = self.hiddenWeights * 2 - 1
            if self.ORTH:
                self.hiddenWeights = orthogonalization(self.hiddenWeights)

    def reset(self):
        self.H = self.initial_H

    def train(self, features, targets, RESETTING=False):
        """
        Step 2: Sequential learning phase
        :param features feature matrix with dimension (numSamples, numInputs)
        :param targets target matrix with dimension (numSamples, numOutputs)
        """
        (numSamples, numOutputs) = targets.shape
        assert features.shape[0] == targets.shape[0]

        H = self.calculateHiddenLayerActivation(features)
        Ht = np.transpose(H)
        h = H[0]
        ht = np.transpose(h)
        target = targets[0]
        self.error = np.transpose(target) - np.dot(ht, (self.forgettingFactor) * self.beta)

        try:
            # update forgetting factor
            self.forgettingFactor = self.FFmin + (1 - self.FFmin) * math.exp(-6 * abs((self.error)))

            if (self.forgettingFactor < self.FFmin):
                self.forgettingFactor = self.FFmin

            if (self.forgettingFactor > self.FFmax):
                self.forgettingFactor = self.FFmax

            scale = 1 / (self.forgettingFactor)

            self.M = scale * self.M - np.dot(scale * self.M,
                                             np.dot(Ht, np.dot(
                                                 pinv(np.eye(numSamples) + np.dot(H, np.dot(scale * self.M, Ht))),
                                                 np.dot(H, scale * self.M))))

            self.beta = (self.forgettingFactor) * self.beta + np.dot(self.M, np.dot(Ht, targets - np.dot(H, (
                self.forgettingFactor) * self.beta)))

            if RESETTING:
                beforeTrace = self.trace
                self.trace = self.M.trace()
                print np.abs(beforeTrace - self.trace)
                if np.abs(beforeTrace - self.trace) < self.thresReset:
                    print self.M
                    eig, _ = np.linalg.eig(self.M)
                    lambMin = min(eig)
                    lambMax = max(eig)
                    # lamb = (lambMax+lambMin)/2
                    lamb = lambMax
                    lamb = lamb.real
                    self.M = lamb * np.eye(self.numHiddenNeurons)
                    print "reset"
                    print self.M


        except np.linalg.linalg.LinAlgError:
            print "SVD not converge, ignore the current training cycle"
        # else:
        #   raise RuntimeError

    def predict(self, features):
        """
        Make prediction with feature matrix
        :param features: feature matrix with dimension (numSamples, numInputs)
        :return: predictions with dimension (numSamples, numOutputs)
        """
        H = self.calculateHiddenLayerActivation(features)
        prediction = np.dot(H, self.beta)
        return prediction
