import matplotlib.pyplot as plt
import sys
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from pythonCode.ImageClass import ImageClass
from itertools import islice
import re
import scipy.signal as sig

class ImageCorrelation:

    #
    # 1. 1.	Create a function for comparing each test image with the collection of training images, using the correlation
    # with each training image without offset (i.e. calculate the sum of the element-wise multiplied images).
    #
    def correlate(self, image1, image2):
        image1Shape = image1.shape
        image2Shape = image2.shape
        correlation = 0
        for y in range(0, image1Shape[0]):
            for x in range(0, image1Shape[1]):
                correlation = correlation + (image1[y,x] * image2[y,x])
        return correlation

    #
    # 2. Estimate the most likely label for each test image by matching with the training image that has maximal correlation.
    # correlate all images in imagesArray to the singleImage.
    # return the index (in imagesArray) of the image which has the
    # largest correlation value
    #
    def correlateImagesToSingleImage(self, singleTrainingImage, testImagesArray):
        correlation = []
        for imageIndex in range(0, len(testImagesArray)):
            correlation.append(self.correlate(singleTrainingImage, testImagesArray[imageIndex]))
        maxCorrelation = max(correlation)
        maxCorrelationImageIndex = correlation.index(maxCorrelation)
        return maxCorrelationImageIndex


    #

    def correlateTrainingImagesToTestImages(self, trainingImages, testImages):
        image2CorrelationIndeces = []
        for imageIndex in range(0, len(trainingImages)):
            image2CorrelationIndeces.append(
                self.correlateImagesToSingleImage(trainingImages[imageIndex], testImages))

        return image2CorrelationIndeces

    # 3.	Estimate the accuracy of your system, as a fraction of the correctly estimated test labels over the total
    # number of test images.
    def calculateCorrelationAccuracy(self, testImagesCorrelationIndeces, trainingLabels, testLabels):
        correctlyEstimatedCount = 0
        for labelsIndex in range(0, len(testImagesCorrelationIndeces)  ):
            if trainingLabels[labelsIndex] == testLabels[testImagesCorrelationIndeces[labelsIndex]]:
                correctlyEstimatedCount =  correctlyEstimatedCount + 1
        return correctlyEstimatedCount / len(trainingLabels)

Test = True
if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    Test = False
if Test:
    plt.rcParams['figure.figsize'] = (13, 6)
    trainImages = ImageClass()
    trainImages.readImages('../data/digits-training.txt')
    testImages = ImageClass()
    testImages.readImages('../data/digits-test.txt')
    correlator = ImageCorrelation()
    index = correlator.correlateImagesToSingleImage(trainImages.digitPixels[0], testImages.digitPixels)
    print("Matched index = ", index)
    testImageCorrelationIncedeces = correlator.correlateTrainingImagesToTestImages(trainImages.digitPixels, testImages.digitPixels)
    accuracy = correlator.calculateCorrelationAccuracy(testImageCorrelationIncedeces, trainImages.digitLabels, testImages.digitLabels)
    print("Correlation accuracy: ", accuracy)
    # takes a long time, got 0.8479834539813857, not bad

