import parse 		#as parse
import dnn 			#as dnn
import labelUtil
import time

TRAIN_FEATURE_FILENAME = "MLDS_HW1_RELEASE_v1/fbank/train.ark"  #_fbank_10000
TRAIN_LABEL_FILENAME = "MLDS_HW1_RELEASE_v1/label/train.lab"
TEST_FEATURE_FILENAME = "MLDS_HW1_RELEASE_v1/fbank/test.ark"
SAVE_MODEL_FILENAME = None#"models/dnn.model"
LOAD_MODEL_FILENAME = None#"models/DNN_ER624_CO0.76426_HL256-3_EP3_LR0.25_BS256.model"
OUTPUT_CSV_FILE_NAME = "output/result.csv"

HIDDEN_LAYER = [128, 128, 128]
LEARNING_RATE_INIT = 0.5
LEARNING_RATE_DECAY = 0.75
EPOCH_NUM = 1
BATCH_SIZE = 256

BRANCH_NUM = 2

curEpoch = 0
learningRates = [LEARNING_RATE_INIT, LEARNING_RATE_INIT*LEARNING_RATE_DECAY]
errorRates = [1.0] * BRANCH_NUM
modelNames = [None] * BRANCH_NUM
loadModelName = None

print 'Parsing...'
t0 = time.time()
trainFeats, trainLabels, trainFrameNames = parse.parseTrainData(TRAIN_FEATURE_FILENAME, TRAIN_LABEL_FILENAME)
testFeats, testFrameNames = parse.parseTestData(TEST_FEATURE_FILENAME)
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

NEURON_NUM_LIST = [ len(trainFeats[0]) ] + HIDDEN_LAYER + [ labelUtil.LABEL_NUM ]

while True:
    for branchIndex in xrange(BRANCH_NUM):
        print 'Training...'
        aDNN = dnn.dnn( NEURON_NUM_LIST, learningRates[branchIndex], EPOCH_NUM, BATCH_SIZE, loadModelName )

        #print 'Saving Neural Network Model...'
        #aDNN.saveNeuralNetwork(OUTPUT_MODEL_FILENAME)
        t2 = time.time()
        aDNN.train(trainFeats, trainLabels)
        t3 = time.time()
        print '...costs ', t3 - t2, ' seconds'
        #print aDNN.errorNum
        print 'Error rate: ', aDNN.errorRate

        modelInfo = "_ER" + str(aDNN.errorRate)[2:5] \
            + "_CO" + str(aDNN.cost)[0:7] \
            + "_HL" + str(HIDDEN_LAYER[0]) + "-" + str(len(HIDDEN_LAYER)) \
            + "_EP" + str(curEpoch) \
            + "_LR" + str(curLearningRate) \
            + "_BS" + str(BATCH_SIZE)
        SAVE_MODEL_FILENAME = "models/DNN" + modelInfo + ".model"
        aDNN.saveModel(SAVE_MODEL_FILENAME)

        print 'Testing...'
        t4 = time.time()
        testLabels = aDNN.test(testFeats)
        t5 = time.time()
        print '...costs', t5 - t4, ' seconds'

        print 'Writing to csv file...'
        OUTPUT_CSV_FILE_NAME = "output/TEST" + modelInfo + ".csv"
        parse.outputTestLabelAsCsv(testFrameNames, testLabels, OUTPUT_CSV_FILE_NAME)

    """
    startNewStr = raw_input('\nstart a new training? (Y/n) ')
    if startNewStr == 'n' or startNewStr == 'N':
        break

    print '    current learning rate: ', LEARNING_RATE
    inputLrStr = raw_input('        new learning rate: ')
    if not not inputLrStr:
        LEARNING_RATE = float(inputLrStr)

    print '    current batch size: ', BATCH_SIZE
    inputBsStr = raw_input('        new batch size: ')
    if not not inputBsStr:
        BATCH_SIZE = int(inputBsStr)

    print '    current epoch num: ', EPOCH_NUM
    inputEnStr = raw_input('        new epoch num: ')
    if not not inputEnStr:
        EPOCH_NUM = int(inputEnStr)
    """

    curEpoch += EPOCH_NUM
