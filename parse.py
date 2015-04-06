
def parseTrainData(TRAIN_FEATURE_FILENAME, TRAIN_LABEL_FILENAME):

    #TRAIN_FEATURE_FILENAME = "MLDS_HW1_RELEASE_v1/fbank/train_fbank_10000.ark"
    #TRAIN_LABEL_FILENAME = "MLDS_HW1_RELEASE_v1/label/train.lab"

    trainFeats = []
    trainLabels = []
    trainFrameNames = []
    #trainFeatCount = 0
    #labelCount = 0
    #trainFeatDim = 0    #dimension

    #parse training features
    with open(TRAIN_FEATURE_FILENAME) as trainFeatFile:
        for line in trainFeatFile:
            if line.rstrip():
                lineList = line.rstrip().split(" ")
                trainFrameNames.append( lineList.pop(0) )
                trainFeats.append( [ float(ele) for ele in lineList ] )
                #trainFeatCount += 1

    #parse training labels
    with open(TRAIN_LABEL_FILENAME) as trainLabelFile:
        for line in trainLabelFile:
            if line.rstrip():
                lineList = line.rstrip().split(",")
                trainLabels.append(lineList[1])
                #labelCount += 1

#    if not not trainFeats:
#        trainFeatDim = len(trainFeats[0])
    
    return (trainFeats, trainLabels, trainFrameNames)



def parseTestData(TEST_FEATURE_FILENAME):

    #TEST_FEATURE_FILENAME = "MLDS_HW1_RELEASE_v1/fbank/test.ark"

    testFeats = []
    testFrameNames = []
    #testFeatCount = 0
    #testFeatDim = 0

    #parse testing features
    with open(TEST_FEATURE_FILENAME) as testFeatFile:
        for line in testFeatFile:
            if line.rstrip():
                lineList = line.rstrip().split(" ")
                testFrameNames.append( lineList.pop(0) )
                testFeats.append( [ float(ele) for ele in lineList ] )
                #testFeatCount += 1

#    if not not testFeats:
#        testFeatDim = len(testFeats[0])
    
    return (testFeats, testFrameNames)

