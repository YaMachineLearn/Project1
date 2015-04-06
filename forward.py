import numpy
import theano
import theano.tensor as T

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
def ForwardPropagate(inputVector, weightMatrices, biasArrays):
	layerNum = len(weightMatrices)
	lineIn = numpy.asarray(inputVector, dtype=theano.config.floatX)
	for i in range(layerNum):
		weightMatrix = numpy.asarray(weightMatrices[i], dtype=theano.config.floatX)
		biasVector = numpy.asarray(biasArrays[i], dtype=theano.config.floatX)
		lineOut = line_out(lineIn, weightMatrix, biasVector)
		lineIn = sigmoid(lineOut)  # the output of the current layer is the input of the next layer
	return lineIn