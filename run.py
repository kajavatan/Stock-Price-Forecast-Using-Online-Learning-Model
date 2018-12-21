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

import csv
import time
from optparse import OptionParser
from matplotlib import pyplot as plt
from scipy import random
import pandas as pd
import numpy as np
from algorithms.OS_ELM import OSELM
from algorithms.OR_ELM import ORELM
from algorithms.OR_ELM_AFF import ORELM_AFF

def _getArgs():
    parser = OptionParser(usage="%prog [options]")
    parser.add_option("-d",
                      "--dataSet",
                      type=str,
                      default='Nikkei',
                      dest="dataSet",
                      help="DataSet Name, choose from sine or nyc_taxi")
    parser.add_option("-l",
                      "--numLags",
                      type=int,
                      default='171',
                      help="the length of time window, this is used as the input dimension of the network")
    parser.add_option("-n",
                      "--numNeurons",
                      type=int,
                      default='69',
                      help="the number of neuron in the hidden layer")
    parser.add_option("-p",
                      "--predStep",
                      type=int,
                      default='1',
                      help="the prediction step of the output")
    parser.add_option("-a",
                      "--algorithm",
                      type=str,
                      default='ORELM-AFF-GA',
                      help="Algorithm name, choose from OSELM, ORELM and ORELM-AFF-GA")
    (options, remainder) = parser.parse_args()
    return options, remainder


def initializeNet(nDimInput, nDimOutput, min_t, max_t, numNeurons, algorithm='ORELM',
                  LN=True, InWeightFF=0.999, OutWeightFF=0.999, HiddenWeightFF=0.999,
                  ORTH=True, AE=True, PRINTING=True):
    assert algorithm == 'OSELM' or algorithm == 'ORELM' or algorithm == 'ORELM-AFF-GA'

    if algorithm == 'OSELM':
        '''
        OSELM
        '''
        net = OSELM(nDimInput, nDimOutput,
                    numHiddenNeurons=numNeurons,
                    activationFunction='sig')

    elif algorithm == 'ORELM':
        '''
        Online Recurrent Extreme Learning Machine (OR-ELM).
        FOSELM + layer normalization + forgetting factor + input layer weight auto-encoding + hidden layer weight auto-encoding.
        '''
        net = ORELM(nDimInput, nDimOutput,
                    numHiddenNeurons=numNeurons,
                    activationFunction='sig',
                    LN=LN,
                    inputWeightForgettingFactor=InWeightFF,
                    outputWeightForgettingFactor=OutWeightFF,
                    hiddenWeightForgettingFactor=HiddenWeightFF,
                    ORTH=ORTH,
                    AE=AE)

    elif algorithm == 'ORELM-AFF-GA':
        '''
        Online Recurrent Extreme Learning Machine (OR-ELM) with forgetting factor
        FOSELM + layer normalization + forgetting factor + input layer weight auto-encoding + hidden layer weight auto-encoding.
        '''
        net = ORELM_AFF(nDimInput, nDimOutput,
                    numHiddenNeurons=numNeurons,
                    activationFunction='sig',
                    min_t=min_t,
                    max_t=max_t,
                    LN=True,
                    inputWeightForgettingFactor=InWeightFF,
                    outputWeightForgettingFactor=OutWeightFF,
                    hiddenWeightForgettingFactor=HiddenWeightFF,
                    ORTH=ORTH,
                    AE=AE)

    return net


def readDataSet(dataSet):
    filePath = 'data/' + dataSet + '.csv'

    if dataSet == 'MackeyGlass':
        df = pd.read_csv(filePath, header=0, skiprows=[1, 2])
        sequence = df['y']

        seq = pd.DataFrame(np.array(pd.concat([sequence], axis=1)),
                           columns=['data'])
    elif dataSet == 'Nikkei':
        df = pd.read_csv(filePath, header=0)
        sequence = df['Close']

        seq = pd.DataFrame(np.array(pd.concat([sequence], axis=1)),
                           columns=['data'])

    else:
        raise (' unrecognized dataset type ')

    return seq


def getTimeEmbeddedMatrix(sequence, numLags, predictionStep=1):
    print "generate time embedded matrix "
    inDim = numLags
    X = np.zeros(shape=(len(sequence), inDim))
    T = np.zeros(shape=(len(sequence), 1))
    for i in xrange(numLags - 1, len(sequence) - predictionStep):
        X[i, :] = np.array(sequence['data'][(i - numLags + 1):(i + 1)])
        T[i, :] = sequence['data'][i + predictionStep]
    return (X, T)

def saveResultToFile(dataSet, predictedInput, algorithmName,predictionStep):
  inputFileName = 'data/' + dataSet + '.csv'
  inputFile = open(inputFileName, "rb")
  csvReader = csv.reader(inputFile)
  # skip header rows
  csvReader.next()
  outputFileName = './prediction/' + dataSet + '_' + algorithmName + '_pred.csv'
  outputFile = open(outputFileName, "w")
  csvWriter = csv.writer(outputFile)
  csvWriter.writerow(
    ['timestamp', 'data', 'prediction-' + str(predictionStep) + 'step'])
  csvWriter.writerow(['datetime', 'float', 'float'])
  csvWriter.writerow(['', '', ''])

  for i in xrange(len(sequence)):
    row = csvReader.next()
    csvWriter.writerow([row[0], row[1], predictedInput[i]])

  inputFile.close()
  outputFile.close()
  print 'Prediction result is saved to ' + outputFileName

if __name__ == "__main__":

    (_options, _args) = _getArgs()
    algorithm = _options.algorithm
    dataSet = _options.dataSet
    numLags = _options.numLags
    predictionStep = _options.predStep
    numNeurons = _options.numNeurons

    print "run ", algorithm, " on ", dataSet
    start = time.time()
    # prepare dataset
    sequence = readDataSet(dataSet)
    # standardize data by subtracting mean and dividing by std
    meanSeq = np.mean(sequence['data'])
    stdSeq = np.std(sequence['data'])
    sequence['data'] = (sequence['data'] - meanSeq) / stdSeq

    (X, T) = getTimeEmbeddedMatrix(sequence, numLags, predictionStep)

    random.seed(6)

    net = initializeNet(nDimInput=X.shape[1],
                        nDimOutput=1,
                        numNeurons=numNeurons,
                        algorithm=algorithm,
                        min_t=numLags,
                        max_t=X.shape[0],
                        LN=True,
                        InWeightFF=0.999,
                        OutWeightFF=0.990,
                        HiddenWeightFF=0.999,
                        AE=True,
                        ORTH=False)

    predictedInput = np.zeros((len(sequence),))
    targetInput = np.zeros((len(sequence),))
    trueData = np.zeros((len(sequence),))

    if algorithm == 'OSELM':
        net.initializePhase(X, T)
    else:
        net.initializePhase(lamb=0.0001)

    for i in xrange(numLags, len(sequence) - predictionStep - 1):
        net.train(X[[i], :], T[[i], :])
        Y = net.predict(X[[i + 1], :])

        predictedInput[i + 1] = Y[-1]
        targetInput[i + 1] = sequence['data'][i + 1 + predictionStep]
        trueData[i + 1] = sequence['data'][i + 1]
        print "{:5}th timeStep -  target: {:8.4f}   |    prediction: {:8.4f} ".format(i, targetInput[i + 1],
                                                                                      predictedInput[i + 1])
        if Y[-1] > 5:
            print "Output has diverged, terminate the process"
            predictedInput[(i + 1):] = 10000
            break

    end = time.time()
    print("Time {}".format(end - start))

    '''
    Calculate total Normalized Root Mean Square Error (NRMSE)
    '''
    # Reconstruct original value
    predictedInput = (predictedInput * stdSeq) + meanSeq
    targetInput = (targetInput * stdSeq) + meanSeq
    trueData = (trueData * stdSeq) + meanSeq
    # Calculate NRMSE from stpTrain to the end
    skipTrain = numLags
    from plot import computeSquareDeviation

    squareDeviation = computeSquareDeviation(predictedInput, targetInput)
    squareDeviation[:skipTrain] = None
    nrmse = np.sqrt(np.nanmean(squareDeviation)) / np.nanstd(targetInput)
    print "NRMSE {}".format(nrmse)

    '''
    Calculate Mean Absolute Percent Error (MAPE)
    '''
    mape = np.mean(np.abs((targetInput - predictedInput) / targetInput)) * 100
    print "MAPE {}".format(mape)
    # 'FF' + str(net.forgettingFactor)+ str(net.numHiddenNeurons)
    # Save prediction result as csv file
    saveResultToFile(dataSet, predictedInput, algorithm, predictionStep)

    '''
    Plot predictions and target values
    '''
    plt.figure(figsize=(15, 6))
    targetPlot, = plt.plot(targetInput, label='target', color='black', marker='.', linestyle='-', linewidth=0.7)
    predictedPlot, = plt.plot(predictedInput, label='predicted', color='#b71653', marker='.', linestyle=':',
                              linewidth=0.7)
    if dataSet == 'nyc_taxi':
        plt.xlim([13000, 13500])
        plt.ylim([0, 30000])
    elif dataSet == 'google_stock':
        plt.xlim([0, 2000])
        plt.ylim([-5, 5])
    elif dataSet == 'Nikkei':
        plt.xlim([0, 10000])
        plt.ylim([0, 40000])
    elif dataSet == 'MackeyGlass':
        plt.xlim([100, 500])
        plt.ylim([0, 2])
    else:
        raise (' unrecognized dataset type ')

    plt.ylabel('value', fontsize=15)
    plt.xlabel('time', fontsize=15)
    plt.ion()
    plt.grid()
    plt.legend(handles=[targetPlot, predictedPlot])
    plt.title('Time-series Prediction of ' + algorithm + ' on ' + dataSet + ' dataset', fontsize=20, fontweight=40)
    plot_path = './fig/predictionPlot.png'
    plt.savefig(plot_path, plot_pathbbox_inches='tight')
    plt.draw()
    plt.show()
    plt.pause(0)
    print 'Prediction plot is saved to' + plot_path

# Grade function for genetic algorithm
def initializedata():
    dataSet = 'Nikkei'

    # prepare dataset
    sequence = readDataSet(dataSet)
    meanSeq = np.mean(sequence['data'])

    # standardize data
    stdSeq = np.std(sequence['data'])
    sequence['data'] = (sequence['data'] - meanSeq) / stdSeq
    return sequence

def calerror(sequence, numlags, numneurons):
    predictionStep = 1
    numLags = numlags

    (X, T) = getTimeEmbeddedMatrix(sequence, numLags, predictionStep)

    random.seed(6)
    net = initializeNet(nDimInput=X.shape[1],
                        nDimOutput=1,
                        numNeurons=numneurons,
                        algorithm='ORELM',
                        min_t=numLags,
                        max_t=X.shape[0],
                        LN=True,
                        InWeightFF=0.999,
                        OutWeightFF=0.990,
                        HiddenWeightFF=0.999,
                        AE=True,
                        ORTH=False)

    net.initializePhase(lamb=0.0001)

    predictedInput = np.zeros((len(sequence),))
    targetInput = np.zeros((len(sequence),))
    trueData = np.zeros((len(sequence),))
    print 'numLags = {}'.format(numLags)
    print 'numneurons = {}'.format(numneurons)

    for i in xrange(numLags, len(sequence) - predictionStep - 1):
        net.train(X[[i], :], T[[i], :])
        Y = net.predict(X[[i + 1], :])

        predictedInput[i + 1] = Y[-1]
        targetInput[i + 1] = sequence['data'][i + 1 + predictionStep]
        trueData[i + 1] = sequence['data'][i + 1]

        if Y[-1] > 5:
            # print "Output has diverged, terminate the process"
            predictedInput[(i + 1):] = 10000
            break

    meanSeq = np.mean(sequence['data'])
    stdSeq = np.std(sequence['data'])
    # Reconstruct original value
    predictedInput = (predictedInput * stdSeq) + meanSeq
    targetInput = (targetInput * stdSeq) + meanSeq
    trueData = (trueData * stdSeq) + meanSeq
    # Calculate NRMSE from stpTrain to the end
    skipTrain = numLags
    from plot import computeSquareDeviation
    squareDeviation = computeSquareDeviation(predictedInput, targetInput)
    squareDeviation[:skipTrain] = None
    nrmse = np.sqrt(np.nanmean(squareDeviation)) / np.nanstd(targetInput)

    return nrmse
