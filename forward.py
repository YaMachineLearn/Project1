import numpy as np
import theano
import theano.tensor as T

from theano import shared
from theano import function

## Theano symbolic variable declarations
inputVector = T.matrix('inputVector')
weightMatrix = T.matrix('weightMatrix')
biasVector = T.matrix('biasVector')
lineOutput = T.matrix('lineOutput')
activationOutput = T.matrix('activationOutput')

## Theano symbolic expression graph
lineOutput = T.dot(weightMatrix, inputVector) + biasVector
activationOutput = 1 / (1 + T.exp(-lineOutput))

## Theano Compilation
line_out = function([inputVector, weightMatrix, biasVector], lineOutput)
sigmoid = function([lineOutput], activationOutput)

## Forward Propagation Function
def getForwardFunction(testFeats, batchSize, weightMatrices, biasArrays):
    index = T.iscalar()
    testFeatsArray = shared(np.transpose(np.asarray(testFeats, dtype=theano.config.floatX)))
    inputVectorArray = testFeatsArray[:, index * batchSize:(index + 1) * batchSize]
    lineIn = inputVectorArray
    for i in range( len(weightMatrices) ):
        weightMatrix = weightMatrices[i]
        biasVector = biasArrays[i]
        lineOutput = T.dot(weightMatrix, lineIn) + T.extra_ops.repeat(biasVector, batchSize, 1)
        lineIn = 1. / (1. + T.exp(-lineOutput)) # the output of the current layer is the input of the next layer
    outputVectorArray = lineIn
    test_model = function(inputs=[index], outputs=outputVectorArray)
    return test_model

def ForwardPropagate(inputVector, weightMatrices, biasArrays):
    layerNum = len(weightMatrices)
    lineIn = np.asarray(inputVector, dtype=theano.config.floatX)
    for i in range(layerNum):
        weightMatrix = np.asarray(weightMatrices[i], dtype=theano.config.floatX)
        biasVector = np.asarray(biasArrays[i], dtype=theano.config.floatX)
        lineOut = line_out(lineIn, weightMatrix, biasVector)
        lineIn = sigmoid(lineOut)  # the output of the current layer is the input of the next layer
    return lineIn