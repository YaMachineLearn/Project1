
import forward
import labelUtil
import theano
import numpy as np
import theano.tensor as T
from theano import shared
from theano import function

class dnn:
    def __init__(self, neuronNumList, learningRate, epochNum):
        self.neuronNumList = neuronNumList    #ex: [128, 128, 128]
        self.learningRate = learningRate
        self.epochNum = epochNum
        self.neuronNumList = neuronNumList

        #better: check if input variables are in right format
        
        #neuronNumList: [69, 128, 128, 128, 48]
        self.weightMatrices = []
        self.biasArrays = []        
        for i in range( len(self.neuronNumList)-1 ):    #ex: range(5-1) => 0, 1, 2, 3
            self.weightMatrices.append( shared( np.asarray( np.random.normal(
                loc=0.0, scale=1.0/np.sqrt(self.neuronNumList[i]),
                size=(self.neuronNumList[i+1], self.neuronNumList[i])), dtype=theano.config.floatX) ) )
            self.biasArrays.append( shared( np.asarray( np.random.normal(
                loc=0.0, scale=1.0/np.sqrt(self.neuronNumList[i]),
                size=(self.neuronNumList[i+1], 1)), dtype=theano.config.floatX) ) )
        #ex: weightMatrices == [ [ [,,],[,,],...,[,,] ], [ [,,],[,,],...,[,,] ], ... ]
        #ex: biasArrays == [ [ [0],[0],...,[0] ], [ [0],[0],...,[0] ], ... ]
        
    def train(self, trainFeats, trainLabels):
        index = T.iscalar()
        trainFeatsArray = shared(np.transpose(np.asarray(trainFeats, dtype=theano.config.floatX)))
        inputVector = trainFeatsArray[:,[index]]
        trainLabelsArray = shared(np.transpose(labelUtil.labelToArray(trainLabels)))
        outputVectorRef = trainLabelsArray[:,[index]]
        lineIn = inputVector
        for i in range( len(self.weightMatrices) ):
            weightMatrix = self.weightMatrices[i]
            biasVector = self.biasArrays[i]
            lineOutput = T.dot(weightMatrix, lineIn) + biasVector
            lineIn = 1. / (1. + T.exp(-lineOutput)) # the output of the current layer is the input of the next layer
        outputVector = lineIn
        cost = T.sum(T.sqr(outputVector - outputVectorRef))
        params = self.weightMatrices + self.biasArrays
        gparams = [T.grad(cost, param) for param in params]
        updates = [
            (param, param - self.learningRate * gparam) # (old parameters, updated parameters)
            for param, gparam in zip(params, gparams)
        ]
        train_model = function(inputs=[index], outputs=[outputVector, cost], updates=updates)
        
        for epoch in xrange(self.epochNum):
            for i in xrange(10): # number of frames to be input
                self.out, self.cost = train_model(i)

        self.errorNum = 0
        for i in xrange(10): # number of frames to be input
            self.out, self.cost = train_model(i)
            if ( T.argmax(self.out).eval() != labelUtil.DICT_LABEL_NUM[trainLabels[i]] ):
                self.errorNum = self.errorNum + 1
        self.errorRate = self.errorNum / 10.0 # number of frames to be input

        #self.out = []
        #for i in xrange(10):
        #    self.out.append(train_model(i))

        #self.outputVector = forward.ForwardPropagate(shared(np.asarray(trainFeats[0], dtype=theano.config.floatX)), self.weightMatrices, self.biasArrays)

    def test(self, testFeats):
        index = T.iscalar()
        testFeatsArray = shared(np.transpose(np.asarray(testFeats, dtype=theano.config.floatX)))
        inputVector = testFeatsArray[:,[index]]
        lineIn = inputVector
        for i in range( len(self.weightMatrices) ):
            weightMatrix = self.weightMatrices[i]
            biasVector = self.biasArrays[i]
            lineOutput = T.dot(weightMatrix, lineIn) + biasVector
            lineIn = 1. / (1. + T.exp(-lineOutput)) # the output of the current layer is the input of the next layer
        outputVector = lineIn
        test_model = function(inputs=[index], outputs=outputVector)
        
        testLabels = []
        for i in xrange(3):
            testLabels.append(test_model(i))
        return testLabels

"""
class neuronLayer:
    def __init__(self, prevNeuronNum, thisNeuronNum):
        self.prevNeuronNum = prevNeuronNum
        self.thisNeuronNum = thisNeuronNum

        self.weights = [ [0] * prevNeuronNum ] * thisNeuronNum
        self.biases = [0] * thisNeuronNum
"""
