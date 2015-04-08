import parse 		#as parse
import dnn 			#as dnn
import labelUtil
import time
import math

TRAIN_FEATURE_FILENAME = "MLDS_HW1_RELEASE_v1/fbank/train.ark"  #_fbank_10000
TRAIN_LABEL_FILENAME = "MLDS_HW1_RELEASE_v1/label/train.lab"
TEST_FEATURE_FILENAME = "MLDS_HW1_RELEASE_v1/fbank/test.ark"
SAVE_MODEL_FILENAME = None#"models/dnn.model"
LOAD_MODEL_FILENAME = None#"models/DNN_ER624_CO0.76426_HL256-3_EP3_LR0.25_BS256.model"
OUTPUT_CSV_FILE_NAME = "output/result.csv"

HIDDEN_LAYER = [256, 256, 256]
LEARNING_RATE = 0.5
EPOCH_NUM = 1
BATCH_SIZE = 256

#params for first epoch
INIT_MODEL_NUM = 8
errorRates = [1.0] * INIT_MODEL_NUM
modelNames = [None] * INIT_MODEL_NUM



print 'Parsing...'
t0 = time.time()
trainFeats, trainLabels, trainFrameNames = parse.parseTrainData(TRAIN_FEATURE_FILENAME, TRAIN_LABEL_FILENAME)
testFeats, testFrameNames = parse.parseTestData(TEST_FEATURE_FILENAME)
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

NEURON_NUM_LIST = [ len(trainFeats[0]) ] + HIDDEN_LAYER + [ labelUtil.LABEL_NUM ]

for initModelIndex in xrange(INIT_MODEL_NUM):

    print '\nTraining...'
    aDNN = dnn.dnn( NEURON_NUM_LIST, LEARNING_RATE, EPOCH_NUM, BATCH_SIZE )

    #print 'Saving Neural Network Model...'
    #aDNN.saveNeuralNetwork(OUTPUT_MODEL_FILENAME)
    t2 = time.time()
    aDNN.train(trainFeats, trainLabels)
    t3 = time.time()
    print '...costs ', t3 - t2, ' seconds'
    #print aDNN.errorNum
    print 'Error rate: ', aDNN.errorRate

    #update model info
    errorRates[initModelIndex] = aDNN.errorRate
    modelInfo = ( "_ER" + str( round(aDNN.errorRate*100000)/100000.0 ) +
        "_CO" + str( round(aDNN.cost*100000)/100000.0 ) +
        "_HL" + str(HIDDEN_LAYER[0]) + "-" + str(len(HIDDEN_LAYER)) +
        "_EP" + str(EPOCH_NUM) +
        "_LR" + str( round(LEARNING_RATE*100000)/100000.0 ) +
        "_BS" + str(BATCH_SIZE) )
    modelNames[initModelIndex] = "models/DNN" + modelInfo + ".model"
    aDNN.saveModel(modelNames[initModelIndex])

bestModelIndex = 0
minErrorRate = errorRates[0]
for modelIndex in xrange(INIT_MODEL_NUM):
    if errorRates[modelIndex] < minErrorRate:
        bestModelIndex = modelIndex
        minErrorRate = errorRates[modelIndex]

bestModelName = modelNames[bestModelIndex]

print "\nmin error rate:", minErrorRate
print "best model name:", bestModelName