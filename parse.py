
def parseTrainData(TRAIN_FEATURE_FILENAME, TRAIN_LABEL_FILENAME):

    #TRAIN_FEATURE_FILENAME = "MLDS_HW1_RELEASE_v1/fbank/train_fbank_10000.ark"
    #TRAIN_LABEL_FILENAME = "MLDS_HW1_RELEASE_v1/label/train.lab"

    trainFeats = []
    trainLabels = []
    trainFrameNames = []
    #trainFeatDim = 0    #dimension

    #parse training features
    with open(TRAIN_FEATURE_FILENAME) as trainFeatFile:
        for line in trainFeatFile:
            if line.rstrip():
                lineList = line.rstrip().split(" ")
                trainFrameNames.append( lineList.pop(0) )
                trainFeats.append( [ float(ele) for ele in lineList ] )
    trainFeatFile.close()

    #parse training labels
    with open(TRAIN_LABEL_FILENAME) as trainLabelFile:
        for line in trainLabelFile:
            if line.rstrip():
                lineList = line.rstrip().split(",")
                trainLabels.append(lineList[1])
    trainLabelFile.close()

    #if not not trainFeats:
    #    trainFeatDim = len(trainFeats[0])
    
    return (trainFeats, trainLabels, trainFrameNames)



def parseTestData(TEST_FEATURE_FILENAME):

    #TEST_FEATURE_FILENAME = "MLDS_HW1_RELEASE_v1/fbank/test.ark"

    testFeats = []
    testFrameNames = []

    #parse testing features
    with open(TEST_FEATURE_FILENAME) as testFeatFile:
        for line in testFeatFile:
            if line.rstrip():
                lineList = line.rstrip().split(" ")
                testFrameNames.append( lineList.pop(0) )
                testFeats.append( [ float(ele) for ele in lineList ] )

    #if not not testFeats:
    #    testFeatDim = len(testFeats[0])
    
    return (testFeats, testFrameNames)

def outputTestLabelAsCsv(testFrameNames, testLabels, TEST_CSV_FILE_NAME):
    #TEST_CSV_FILE_NAME = "result.csv"
    #testFrameNames = ['fadg0_si1279_1', 'fadg0_si1279_2', ...]
    #testLabels = ['sil', 'aa', ...]
    with open(TEST_CSV_FILE_NAME, 'w') as testCsvFile:
        testCsvFile.write("Id,Prediction\n")
        for i in xrange( len(testFrameNames) ):
            testCsvFile.write(testFrameNames[i] + ',' + testLabels[i] + '\n')
