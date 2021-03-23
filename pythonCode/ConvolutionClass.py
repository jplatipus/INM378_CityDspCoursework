import sys

from scipy.io import wavfile
from scipy import fft, signal
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# Class that applies a filter to a sample
from pythonCode.WavClass import WavClass


class ConvolutionClass:
    # wav : WavClass
    # filteredSamples

    # output of convolution is the length of the signal + the length of the filter kernel - 1
    # W. Smith, Digital signal processing, Chapter 6.
    def convolveInputSide(self, signal, kernel):
        output = np.zeros(len(signal) + len(kernel) - 1)
        for signalIndex in range(len(signal)):
            for kernelIndex in range(len(kernel)):
                output[signalIndex + kernelIndex] = output[signalIndex + kernelIndex] + signal[signalIndex] * kernel[kernelIndex]
        return output[:-(len(kernel) - 1)]

    # output of convolution is the length of the signal + the length of the filter kernel - 1
    # W. Smith, Digital signal processing, Chapter 6.
    def convolveOutputSide(selfself, signal, kernel):
        output = np.zeros(len(signal) + len(kernel) - 1)
        outputRange = range(len(output))
        kernelRange = range(len(kernel))
        for outputIndex in outputRange:
            for kernelIndex in kernelRange:
                if (outputIndex - kernelIndex < 0) or (outputIndex - kernelIndex > (len(signal) - 1)):
                    # don't do anything
                    pass
                else:
                    output[outputIndex] = output[outputIndex] + kernel[kernelIndex] * signal[outputIndex - kernelIndex]
        return output[:-(len(kernel) - 1)]

    def plotTwoSignals(self, signal1, signal2, title):
        plt.figure()
        plt.plot(signal1, 'r', label='Original')
        plt.plot(signal2, 'b', alpha=0.5, label='Filtered')
        plt.title(title)
        plt.legend(loc=4)
        plt.xlabel("Sample Number");
        plt.ylabel("Amplitude");
        plt.show()

# TEST creating the convolution comb filter min max max values which need to be used in the
# interpolation. Plot the impulse response as well as the frequency spectrum of the filter.
Test = True

if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    Test = False
if Test:
    signal = np.arange(0, 1000, 1)
    filterMin = np.zeros(64)
    filterMin[0] = 1
    filterMin[32] = 0.95
    conv = ConvolutionClass()
    convolvedInputSide = conv.convolveInputSide(signal, filterMin)
    convolvedOutputSide = conv.convolveOutputSide(signal, filterMin)
    conv.plotTwoSignals(signal, convolvedInputSide, "Original vs Input Side Convoluton")
    conv.plotTwoSignals(signal, convolvedOutputSide, "Original vs Output Side Convoluton")
    conv.plotTwoSignals(convolvedInputSide, convolvedOutputSide, "Input Side vs Convoluton Side Convoluton")


