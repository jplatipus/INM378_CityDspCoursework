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

from itertools import islice
import re

#
# Class to encapsulate a set of images
#
class ImageClass:

    def __init__(self):
        print("Image class init")

    #
    # load all the images in the given file
    # sets self.digitPixels[] and self.digitLabels[] to the image data and labels
    #
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
    #   display the requiested image
    #
    def displayImage(self,imageIndex):
        plt.figure()
        plt.imshow(self.digitPixels[imageIndex])
        plt.show()
        print('Image of digit {}'.format(self.digitLabels[imageIndex]))

    def displayImages(self, imageIndeces):
        columns = 4
        rows = int(len(imageIndeces) / 4) + int(len(imageIndeces) % 4)
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.6)
        for imageIndex in range(1, len(imageIndeces)):
            ax = fig.add_subplot(rows, columns, imageIndex)
            ax.imshow(self.digitPixels[imageIndex])
            ax.title.set_text("Label {}".format(self.digitLabels[imageIndex]))
        plt.show()

Test = False
if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    Test = False
if Test:
    plt.rcParams['figure.figsize'] = (13, 6)
    trainImages = ImageClass()
    trainImages.readImages('../data/digits-training.txt')
    testImages = ImageClass()
    testImages.readImages('../data/digits-test.txt')
    trainImages.displayImages([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    testImages.displayImages([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

