import parse 		#as parse
import dnn 			#as dnn
import labelUtil
import time
import math

TRAIN_FEATURE_FILENAME = "MLDS_HW1_RELEASE_v1/fbank/train.ark"  #_fbank_10000
TRAIN_LABEL_FILENAME = "MLDS_HW1_RELEASE_v1/label/train.lab"
TEST_FEATURE_FILENAME = "MLDS_HW1_RELEASE_v1/fbank/test.ark"
SAVE_MODEL_FILENAME = None#"models/dnn.model"
LOAD_MODEL_FILENAME = "models/DNN_CO0.77618_HL1024-2_EP1_LR0.125_BS256.model"
OUTPUT_CSV_FILE_NAME = "output/result.csv"

HIDDEN_LAYER = [1024, 1024]
LEARNING_RATE_INIT = 0.125
LEARNING_RATE_DECAY = 0.5
EPOCH_NUM = 1
START_EPOCH = 3    #one-indexed
BATCH_SIZE = 256

#setting of searching
BRANCH_NUM = 1
MAX_EPOCH = 50  # now useless

curEpoch = START_EPOCH
learningRates = [LEARNING_RATE_INIT, LEARNING_RATE_INIT*LEARNING_RATE_DECAY]
errorRates = [1.0] * BRANCH_NUM
costs = [1.0] * BRANCH_NUM
modelNames = [None] * BRANCH_NUM

print 'Parsing...'
t0 = time.time()
trainFeats, trainLabels, trainFrameNames = parse.parseTrainData(TRAIN_FEATURE_FILENAME, TRAIN_LABEL_FILENAME)
#testFeats, testFrameNames = parse.parseTestData(TEST_FEATURE_FILENAME)
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

NEURON_NUM_LIST = [ len(trainFeats[0]) ] + HIDDEN_LAYER + [ labelUtil.LABEL_NUM ]

while True:
    print "\nepoch: ", (curEpoch)

    for branchIndex in xrange(BRANCH_NUM):
        print "learningRate: ", learningRates[branchIndex]
        print 'Training...'
        aDNN = dnn.dnn( NEURON_NUM_LIST, learningRates[branchIndex], EPOCH_NUM, BATCH_SIZE, LOAD_MODEL_FILENAME )

        #print 'Saving Neural Network Model...'
        #aDNN.saveNeuralNetwork(OUTPUT_MODEL_FILENAME)
        t2 = time.time()
        aDNN.train(trainFeats, trainLabels)
        t3 = time.time()
        print '...costs ', t3 - t2, ' seconds'
        #print aDNN.errorNum
        #print 'Error rate: ', aDNN.errorRate

        #update branch info
        """
        errorRates[branchIndex] = aDNN.errorRate
        modelInfo = ( "_ER" + str( round(aDNN.errorRate*100000)/100000.0 ) +
            "_CO" + str( round(aDNN.totalCost*100000)/100000.0 ) +
            "_HL" + str(HIDDEN_LAYER[0]) + "-" + str(len(HIDDEN_LAYER)) +
            "_EP" + str(curEpoch) +
            "_LR" + str( round(learningRates[branchIndex]*100000)/100000.0 ) +
            "_BS" + str(BATCH_SIZE) )
        modelNames[branchIndex] = "models/DNN" + modelInfo + ".model"
        aDNN.saveModel(modelNames[branchIndex])
        """
        costs[branchIndex] = aDNN.totalCost
        modelInfo = ( "_CO" + str( round(aDNN.totalCost*100000)/100000.0 ) +
            "_HL" + str(HIDDEN_LAYER[0]) + "-" + str(len(HIDDEN_LAYER)) +
            "_EP" + str(EPOCH_NUM) +
            "_LR" + str( round(learningRates[branchIndex]*100000)/100000.0 ) +
            "_BS" + str(BATCH_SIZE) )
        modelNames[branchIndex] = "models/DNN" + modelInfo + ".model"
        aDNN.saveModel(modelNames[branchIndex])

    bestBranchIndex = 0
    """
    minErrorRate = errorRates[0]
    for branchIndex in xrange(BRANCH_NUM):
        if errorRates[branchIndex] < minErrorRate:
            bestBranchIndex = branchIndex
            minErrorRate = errorRates[branchIndex]
    """
    minCost = costs[0]
    for branchIndex in xrange(BRANCH_NUM):
        if costs[branchIndex] < minCost:
            bestBranchIndex = branchIndex
            minCost = costs[branchIndex]

    bestLearningRate = learningRates[bestBranchIndex]
    learningRates = [bestLearningRate, bestLearningRate*LEARNING_RATE_DECAY]

    LOAD_MODEL_FILENAME = modelNames[bestBranchIndex]

    curEpoch += EPOCH_NUM
