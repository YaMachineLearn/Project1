import parse 		#as parse
import dnn 			#as dnn
import labelUtil
import time
import math

TEST_FEATURE_FILENAME = "MLDS_HW1_RELEASE_v1/fbank/test.ark"
LOAD_MODEL_FILENAME = "models/DNN_ER0.53517_CO0.68399_HL256-1_EP10_LR0.00781_BS256.model"
#OUTPUT_CSV_FILE_NAME = "output/result.csv"

modelInfo = "_ER0.53517_CO0.68399_HL256-1_EP10_LR0.00781_BS256"
featDim = 69

HIDDEN_LAYER = [256]
LEARNING_RATE = 0.125
EPOCH_NUM = 1
START_EPOCH = 4    #one-indexed
BATCH_SIZE = 256


print 'Parsing...'
t0 = time.time()
testFeats, testFrameNames = parse.parseTestData(TEST_FEATURE_FILENAME)
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

NEURON_NUM_LIST = [ featDim ] + HIDDEN_LAYER + [ labelUtil.LABEL_NUM ]

aDNN = dnn.dnn( NEURON_NUM_LIST, LEARNING_RATE, EPOCH_NUM, BATCH_SIZE, LOAD_MODEL_FILENAME )

print 'Testing...'
t4 = time.time()
testLabels = aDNN.test(testFeats)
t5 = time.time()
print '...costs', t5 - t4, ' seconds'

print 'Writing to csv file...'
OUTPUT_CSV_FILE_NAME = "output/TEST" + modelInfo + ".csv"
parse.outputTestLabelAsCsv(testFrameNames, testLabels, OUTPUT_CSV_FILE_NAME)
