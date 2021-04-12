import sys
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import skimage.transform as imgTransform
import skimage.util as imgUtil
from itertools import islice
import re

#
# Class to encapsulate a set of images: loads a set of digit images and their labels
# self.digitPixels[] holds all the images, each one a 32x32 pixel array
# self.digitLabels[] holds the digit labels that correspond to the images
#
class ImageClass:

    # constructor
    def __init__(self):
        print("Image class init")

    # create a deep copy of this class
    # startIndex, endIndex, optional subset of images to copy to the new ImageClass instance, all are copied by default
    # return the new copy of ImageClass
    def copy(self, startIndex=0, endIndex=-1):
        copy = ImageClass()
        copy.digitPixels = self.digitPixels.copy()
        copy.digitLabels = self.digitLabels.copy()
        if startIndex != 0 or endIndex != -1:
            copy.digitPixels = copy.digitPixels[startIndex:endIndex]
            copy.digitLabels = copy.digitLabels[startIndex:endIndex]
        return copy

    #
    # load all the images in the given file
    # sets self.digitPixels[] and self.digitLabels[] to the image data and labels
    # filename the file that holds the images and labels to read.
    # stores the images in self.digitPixels and labels in self.digitLabels
    def readImages(self, filename):
        with open(filename) as f:
            header = list(islice(f, 21))
            pixel_height = [int(x[1]) for x in map(lambda r: re.match('entheight = (\d+)', r), header) if x][0]
            num_digits = [int(x[1]) for x in map(lambda r: re.match('ntot = (\d+)', r), header) if x][0]
            self.digitPixels = []
            self.digitLabels = []
            for _ in range(num_digits):
                chunk = list(islice(f, pixel_height + 1))
                self.digitPixels.append(np.loadtxt(chunk[:-1]))
                self.digitLabels.append(int(chunk[-1]))

            display("Loaded {} images and {} labels.".format(len(self.digitPixels), len(self.digitLabels)))

    #
    #   display the requested image
    #
    def displayImage(self,imageIndex):
        plt.figure()
        plt.imshow(self.digitPixels[imageIndex])
        plt.show()
        print('Image of digit {}'.format(self.digitLabels[imageIndex]))

    # displays images, 4 to a row.
    # imageIndeces an optional set of inage indeces to display: default is ALL images
    def displayImages(self, imageIndeces=None):
        if imageIndeces == None:
            imageIndeces = range(0, len(self.digitPixels))
        columns = 4
        rows = int(len(imageIndeces) / 4) + int(len(imageIndeces) % 4)
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.6)
        for imageIndex in range(1, len(imageIndeces)):
            ax = fig.add_subplot(rows, columns, imageIndex)
            ax.imshow(self.digitPixels[imageIndex])
            ax.title.set_text("Label {}".format(self.digitLabels[imageIndex]))
            ax.set(yticklabels=[])
            ax.set(xticklabels=[])
        plt.show()
    '''
    #
    #   display the requested image
    #
    def displayImagePixels(self,imagePixels):
        plt.figure()
        plt.imshow(imagePixels)
        plt.show()
        print('Image of digit {}'.format(imagePixels))

    def displayImagesPixels(self, imagesPixels):
        columns = 4
        rows = int(len(imagesPixels) / 4) + int(len(imagesPixels) % 4)
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.6)
        for imageIndex in range(1, len(imagesPixels)):
            ax = fig.add_subplot(rows, columns, imageIndex)
            ax.imshow(imagesPixels[imageIndex])
            ax.title.set_text("Label {}".format(imagesPixels[imageIndex]))
        plt.show()
    '''

    # converts this class' images to negatives of the images
    # returns an instance of this class, as a convenience
    def convertImagesToNegativeImages(self):
        negatives = []
        for originalImage in self.digitPixels:
            negatives.append((originalImage - 1) * -1)
        self.digitPixels = negatives
        return self

    # rotates this class' images anticlockwise by the number of degrees requested
    # angle the number of degrees to rotate the image
    # returns an instance of this class, as a convenience
    def rotateImages(self, angle):
        rotated = []
        for originalImage in self.digitPixels:
            rotated.append(imgTransform.rotate(originalImage, angle, resize=False, center=None))
        self.digitPixels =  rotated
        return self

    # adds noise to all of this class' images
    # randomSeed the random number generator seed to use (default is 101)
    # returns an instance of this class, as a convenience
    def addNoiseToImages(self, randomSeed = 101):
        noisy = []
        for originalImage in self.digitPixels:
            noisy.append(imgUtil.random_noise(originalImage, mode='s&p', seed=randomSeed))
        self.digitPixels = noisy
        return self

    # shift all the images by x and y pixels
    #
    def offSetImages(self, xOffset, yOffset):
        offsetImages = []

        for originalImage in self.digitPixels:
            if yOffset > 0:
                originalImage = np.pad(originalImage, [(yOffset, 0), (0, 0)], 'constant', constant_values=(0))
                originalImage = originalImage[:-yOffset,:]
            if xOffset > 0:
                originalImage = np.pad(originalImage, [(0, 0), (xOffset, 0)], 'constant', constant_values=(0))
                originalImage = originalImage[:, :-xOffset]
            offsetImages.append(originalImage)
        self.digitPixels = offsetImages
        return self


Test = True
if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    Test = False
if Test:
    plt.rcParams['figure.figsize'] = (13, 6)
    trainImages = ImageClass()
    trainImages.readImages('../data/digits-training.txt')
    testImages = ImageClass()
    testImages.readImages('../data/digits-test.txt')
    #trainImages.displayImages([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    #testImages.displayImages([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    testImages.displayImages([0])

    trainCopy = trainImages.copy(0,4).convertImagesToNegativeImages()
    trainCopy.displayImages()

    trainCopy = trainImages.copy(0, 4).rotateImages(15)
    trainCopy.displayImages()

    trainCopy = trainImages.copy(0, 4).addNoiseToImages(5)
    trainCopy.displayImages()

    trainCopy = trainImages.copy(0,4).displayImages()

    trainCopy = trainImages.copy(0, 4).offSetImages(5,1)
    trainCopy.displayImages()


