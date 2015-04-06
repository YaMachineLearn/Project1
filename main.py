
import parse #as parse
import dnn #as dnn
import labelUtil
import time

TRAIN_FEATURE_FILENAME = "MLDS_HW1_RELEASE_v1/fbank/train_fbank_10000.ark"
TRAIN_LABEL_FILENAME = "MLDS_HW1_RELEASE_v1/label/train_10000.lab"
TEST_FEATURE_FILENAME = "MLDS_HW1_RELEASE_v1/fbank/test.ark"

HIDDEN_LAYER = [128]
LEARNING_RATE = 0.01
EPOCH_NUM = 500

print 'Parsing...'
t0 = time.time()
trainFeats, trainLabels, trainFrameNames = parse.parseTrainData(TRAIN_FEATURE_FILENAME, TRAIN_LABEL_FILENAME)
testFeats, testFrameNames = parse.parseTestData(TEST_FEATURE_FILENAME)
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

NEURON_NUM_LIST = [ len(trainFeats[0]) ] + HIDDEN_LAYER + [ labelUtil.LABEL_NUM ]

print 'Training...'
t2 = time.time()
aDNN = dnn.dnn( NEURON_NUM_LIST, LEARNING_RATE, EPOCH_NUM )
aDNN.train(trainFeats, trainLabels)
t3 = time.time()
print '...costs ', t3 - t2, ' seconds'
#print aDNN.out
#print aDNN.cost
#print aDNN.errorNum
print 'Error rate: ', aDNN.errorRate

#print 'Testing...'
#testLabels = aDNN.test(testFeats)
#print 'testLabels:'
#print testLabels

#outputCSV(testFrameNames, testLabels)
