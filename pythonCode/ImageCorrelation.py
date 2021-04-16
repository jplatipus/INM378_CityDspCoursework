import sys
import matplotlib.pyplot as plt
import numpy as np
from pythonCode.ImageClass import ImageClass
import scipy.signal as sig

class ImageCorrelation:
    #
    # Class that encapsulates the image correlation functionality: 2 correlation versions are available:
    # one is using a simple correlation implementation (correlate()), the other uses signal.correlate2d()
    #
    def __init__(self):
        print("ImageCorrelation init")
    #
    # 1. 1.	Create a function for comparing each test image with the collection of training images, using the correlation
    # with each training image without offset (i.e. calculate the sum of the element-wise multiplied images).
    #
    #  pixels1, pixels2 images pixel arrays to correlate
    # return correlation value
    def correlate(self, pixels1, pixels2):
        # the two images are the same size so the size of one is used
        # to iterate over each pixel of both images
        imageShape = pixels1.shape
        correlation = 0
        for y in range(0, imageShape[0]):
            for x in range(0, imageShape[1]):
                correlation = correlation + (pixels1[y,x] * pixels2[y,x])
        return correlation

    #
    # 2. Estimate the most likely label for each test image by matching with the training image that has maximal correlation.
    # correlate all images in imagesArray to the singleImage.
    # return the index (in imagesArray) of the image which has the
    # largest correlation value
    #
    # singleTrainingDigitPixels training image pixels array
    # testDigitsPixels array of test images pixels array to correlate against
    # return the testImagesArray index of the best match found
    def correlateImagesToSingleImage(self, singleTrainingDigitPixels, testDigitsPixels):
        correlation = []
        # for each test image
        for imageIndex in range(0, len(testDigitsPixels)):
            # calculate the correlation with the singleTrainingImage and save it in correlation[]
            correlation.append(self.correlate(singleTrainingDigitPixels, testDigitsPixels[imageIndex]))
        # find the maximum correlation
        maxCorrelation = max(correlation)
        # using the max correlation value, find the index of it in correlation[]
        maxCorrelationImageIndex = correlation.index(maxCorrelation)
        return maxCorrelationImageIndex


    #
    # correlate using the correlation method in this class all training images with all test images
    #
    # trainingImages class instance with the relevant training images to correlate (in digiPixels)
    # testImages class instance with all the test images
    # return array of best matches as indeces into the test images for each of the training images
    def correlateTrainingImagesToTestImages(self, trainingImages, testImages):
        #print("correlateTrainingImagesToTestImages({}, {})".format(len(trainingImages), len(testImages)))
        image2CorrelationIndeces = []
        # for each training image
        for imageIndex in range(0, len(trainingImages.digitPixels)):
            # find the best match in the test images using correlate(), store the result in image2CorrelationIndeces
            image2CorrelationIndeces.append(
                self.correlateImagesToSingleImage(trainingImages.digitPixels[imageIndex], testImages.digitPixels))
        # return array of best match found in testImages, one perTrainingImages[] entry
        return image2CorrelationIndeces

    # 3. Estimate the accuracy of your system, as a fraction of the correctly estimated test labels over the total
    # number of test images.
    #
    # trainToTestCorrelationIndeces an indeces array of training predictions: each entry contains the index of the
    #                       predicted label in testLabels
    # trainingImages class instance that has an array of the training image labels
    # testImages class instance that has an array of the test labels
    # returns overall accuracy as correctly estimated count / the number of  entries in trainToTestCorrelationIndeces
    def calculateCorrelationAccuracy(self, trainToTestCorrelationIndeces, trainingImages, testImages):
        trainingLabels = trainingImages.digitLabels
        testLabels = testImages.digitLabels
        correctlyEstimatedCount = 0
        ' for each predicted result, count the number of correct matches'
        for labelsIndex in range(0, len(trainToTestCorrelationIndeces)  ):
            testLabelsIndex = trainToTestCorrelationIndeces[labelsIndex]
            if trainingLabels[labelsIndex] == testLabels[testLabelsIndex]:
                correctlyEstimatedCount =  correctlyEstimatedCount + 1
        # return the number of correctly matched training images / the total number of the training images
        return correctlyEstimatedCount / len(trainingLabels)

    #
    # Find the best 2d correlation between the singleTestImage and all the testImages,
    # Use the two-dimensional correlation function signal.correlate2d to find the best match over
    # all image offsets.
    #
    # singleTrainingDigitPixels a single training image digitPixels array
    # testDigitsPixels an array of all the test images digitPixels, each one a 32x32 array
    # return the testImagesArray index with the best correlation
    def calculate2dCorrelation(self, singleTrainingDigitPixels, testDigitsPixels):
        # where each correlation value for each test image is stored, hence the indeces of the correlation values
        # can be used as a key to get the matching test image
        correlations = []
        # for each test image
        for imageIndex in range(0, len(testDigitsPixels)):
            currentTestDigitPixels = testDigitsPixels[imageIndex]
            # correlate test image with training image
            result = sig.correlate2d(singleTrainingDigitPixels, currentTestDigitPixels, mode='same')
            # extract and store in correlations the largest correlation value found
            bestCorrelation = np.amax(result)
            correlations.append(bestCorrelation)
        # find the maximum correlation value
        maxOverallCorrelation = max(correlations)
        # find the index of the max correlation, the index is also valid for the testImages
        maxCorrelationImageIndex = correlations.index(maxOverallCorrelation)
        return maxCorrelationImageIndex

    # Find the best 2d correlation between the singleTestImage and all the testImages,
    # For efficiency you can use a subset of the training images
    #
    # trainingImages training images class instance that holds the relevant training images
    # testImages test image class instance that holds all the test images
    # return the testImagesArray indeces (one per training image) with the best correlation
    def correlate2dTrainingImagesToTestImages(self, trainingImages, testImages):
        image2CorrelationIndeces = []
        # fro each training image
        for imageIndex in range(0, len(trainingImages.digitPixels)):
            # evaluate it best 2d correlation with the test images
            image2CorrelationIndeces.append(
                self.calculate2dCorrelation(trainingImages.digitPixels[imageIndex], testImages.digitPixels))

        return image2CorrelationIndeces

    # perform simple correlation and 2d correlation of the training images against the test images
    # returns 2 values: the overall simple correlation accuracy and the overall 2d correlation accuracy
    def compareImageCorrelations(self, trainingImages, testImages):
        simpleCorrelationIndeces = self.correlateTrainingImagesToTestImages(trainingImages, testImages)
        c2dCorrelationIndeces = self.correlate2dTrainingImagesToTestImages(trainingImages, testImages)
        simpleCorrelationAccuracy = self.calculateCorrelationAccuracy(simpleCorrelationIndeces, trainingImages, testImages)
        c2dCorrelationAccuracy = self.calculateCorrelationAccuracy(c2dCorrelationIndeces, trainingImages, testImages)
        return simpleCorrelationAccuracy, c2dCorrelationAccuracy

Test = True
if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    Test = False
# test loading images
if Test:
    plt.rcParams['figure.figsize'] = (13, 6)
    trainImages = ImageClass()
    trainImages.readImages('../data/digits-training.txt')
    testImages = ImageClass()
    testImages.readImages('../data/digits-test.txt')
    correlator = ImageCorrelation()

if Test:
    print("Quick Test")
    train = trainImages.copy(0, 2)
    test = testImages.copy(0, 2)
    simpleAccuracy, c2dAccuracy = correlator.compareImageCorrelations(train, test)
    print("Accuracies: Corr: {} 2dCorr: {}".format(simpleAccuracy, c2dAccuracy))

if Test:
    print("Negative Images")
    negTrain = trainImages.copy(0,50).convertImagesToNegativeImages()
    negTest =  testImages.copy().convertImagesToNegativeImages()
    simpleAccuracy, c2dAccuracy = correlator.compareImageCorrelations(negTrain, negTest)
    print("Accuracies: Corr: {} 2dCorr: {}".format(simpleAccuracy, c2dAccuracy))

    print("Rotated Images")
    rotTrain = trainImages.copy(0,50).rotateImages(15)
    simpleAccuracy, c2dAccuracy = correlator.compareImageCorrelations(rotTrain, testImages)
    print("Accuracies: Corr: {} 2dCorr: {}".format(simpleAccuracy, c2dAccuracy))

    print("Noisy Images")
    noiseTrain = trainImages.copy(0, 50).addNoiseToImages(5)
    simpleAccuracy, c2dAccuracy = correlator.compareImageCorrelations(noiseTrain, testImages)
    print("Accuracies: Corr: {} 2dCorr: {}".format(simpleAccuracy, c2dAccuracy))

    print("Offset Images")
    offsetTrain = trainImages.copy(0, 50).offSetImages(5, 1)
    simpleAccuracy, c2dAccuracy = correlator.compareImageCorrelations(offsetTrain, testImages)
    print("Accuracies: Corr: {} 2dCorr: {}".format(simpleAccuracy, c2dAccuracy))

    print("Original Images")
    simpleAccuracy, c2dAccuracy = correlator.compareImageCorrelations(trainImages.copy(0, 50), testImages)
    print("Accuracies: Corr: {} 2dCorr: {}".format(simpleAccuracy, c2dAccuracy))