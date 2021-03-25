import sys

from scipy.io import wavfile
from scipy import fft, signal
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# Class that applies a filter to a sample
from pythonCode.ControlClass import ControlClass
from pythonCode.MockControlClass import MockControlClass
from pythonCode.WavClass import WavClass


class ConvolutionClass:

    # Constructor:
    # filter1 is the
    def __init__(self, filter1, filter2, control):
        self.filter1 = filter1
        self.filter2 = filter2
        self.control = control

    # output of convolution is the length of the signal + the length of the filter kernel - 1
    # W. Smith, Digital signal processing, Chapter 6.
    def convolveInputSide(self, signal):
        isStereo = False
        filterLength = len(self.filter1)
        if signal.ndim == 2:
            signalLength = len(signal[:,0])
            output = np.zeros(signalLength + filterLength - 1)
            isStereo = True
            output = np.transpose(np.array([output, output]))
        else:
            signalLength = len(signal)
            output = np.zeros(signalLength + filterLength - 1)


        for signalIndex in range(signalLength):
            for kernelIndex in range(filterLength):
                # create interpolated filter using the control signal as a factor for interpolation
                filter = self.control.interpolateFilters(signalIndex, self.filter1, self.filter2)
                if isStereo:
                    output[signalIndex + kernelIndex, 0] = output[signalIndex + kernelIndex, 0] + signal[signalIndex, 0] * \
                                                        filter[kernelIndex]
                    output[signalIndex + kernelIndex, 1] = output[signalIndex + kernelIndex, 1] + signal[signalIndex, 1] * \
                                                        filter[kernelIndex]
                else:
                    output[signalIndex + kernelIndex] = output[signalIndex + kernelIndex] + signal[signalIndex] * \
                                                        filter[kernelIndex]
        if isStereo:
            return output[:-(len(filter) - 1),:]
        else:
            return output[:-(len(filter) - 1)]

    # output of convolution is the length of the signal + the length of the filter kernel - 1
    # W. Smith, Digital signal processing, Chapter 6.
    def convolveOutputSide(self, signal):
        isStereo = False
        filterLength = len(self.filter1)
        if signal.ndim == 2:
            signalLength = len(signal[:, 0])
            output = np.zeros(signalLength + filterLength - 1)
            isStereo = True
            output = np.transpose(np.array([output, output]))

        else:
            signalLength = len(signal)
            output = np.zeros(signalLength + filterLength - 1)


        outputRange = signalLength + filterLength - 1
        kernelRange = len(self.filter1)
        for outputIndex in range(outputRange):
            for kernelIndex in range(kernelRange):
                filter = self.control.interpolateFilters(outputIndex, self.filter1, self.filter2)
                if (outputIndex - kernelIndex < 0) or (outputIndex - kernelIndex > (len(signal) - 1)):
                    # don't do anything
                    pass
                else:
                    if isStereo:
                        output[outputIndex, 0] = output[outputIndex, 0] + filter[kernelIndex] * signal[
                            outputIndex - kernelIndex, 0]
                        output[outputIndex, 1] = output[outputIndex, 1] + filter[kernelIndex] * signal[
                            outputIndex - kernelIndex, 1]
                    else:
                        output[outputIndex] = output[outputIndex] + filter[kernelIndex] * signal[outputIndex - kernelIndex]
        if isStereo:
            return output[:-(len(self.filter1) - 1), :]
        else:
            return output[:-(len(self.filter1) - 1)]

    # expects 1d signal
    def __plotTwoSignals__(self, signal1, signal2, title):
        plt.figure()
        plt.plot(signal1, 'r', label='Original')
        plt.plot(signal2, 'b', alpha=0.5, label='Filtered')
        plt.title(title)
        plt.legend(loc=4)
        plt.xlabel("Sample Number");
        plt.ylabel("Amplitude");
        plt.show()

    def __plotTwoSignals__(self, signal1, signal2, title):
        plt.figure()
        plt.plot(signal1, 'r', label='Original')
        plt.plot(signal2, 'b', alpha=0.5, label='Filtered')
        plt.title(title)
        plt.legend(loc=4)
        plt.xlabel("Sample Number");
        plt.ylabel("Amplitude");
        plt.show()

    def plotTwoSignals(self, signal1, signal2, title):
        if signal1.ndim == 2:
            title1 = "{} Channel 1".format(title)
            self.__plotTwoSignals__(signal1[:,0], signal2[:,0], title1)
            title2 = "{} Channel 2".format(title)
            self.__plotTwoSignals__(signal1[:, 1], signal2[:, 1], title2)
        else:
            self.__plotTwoSignals__(signal1, signal2, title)

# TEST creating the convolution comb filter min max max values which need to be used in the
# interpolation. Plot the impulse response as well as the frequency spectrum of the filter.
Test = False

if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    Test = False
if Test:
    sig = np.abs(np.sin(2 * np.pi * 0.05 * np.arange(-80, 1)))
    plt.figure()
    plt.plot(sig)
    plt.title("Signal for test")
    plt.show()
    control = ControlClass()
    # in this test there is no convolution
    filter = np.zeros(81)
    filter[0] = 1
    filter[4] = 0.95
    control = MockControlClass()
    control.controlSignal = filter
    conv = ConvolutionClass(filter, filter, control)
    inputSide = conv.convolveInputSide(sig)
    outputSide = conv.convolveOutputSide(sig)
    pythonConv = signal.convolve(filter, sig)
    pythonConv = pythonConv[:(len(filter) - 1)]
    plt.figure()
    plt.plot(inputSide, 'r', label='iSide')
    plt.plot(outputSide, 'b', alpha = 0.5, label='oSide')
    plt.plot(pythonConv, 'g', alpha=0.5, label='python')
    plt.title("PLot of 3 convolution operations: input side, output side and Python's")
    plt.legend()
    plt.show()
    '''
    control = ControlClass()
    display("Testing Mono")
    signal = np.arange(0, 1000, 1)
    filterMin = np.zeros(64)
    filterMin[0] = 1
    filterMin[4] = 0.95
    filterMax = np.zeros(64)
    filterMin[0] = 1
    filterMin[48] = 0.95
    control = ControlClass()
    conv = ConvolutionClass(filterMin, filterMax, control)
    convolvedInputSide = conv.convolveInputSide(signal)
    convolvedOutputSide = conv.convolveOutputSide(signal)
    conv.plotTwoSignals(signal, convolvedInputSide, "Original vs Input Side Convoluton")
    conv.plotTwoSignals(signal, convolvedOutputSide, "Original vs Output Side Convoluton")
    conv.plotTwoSignals(convolvedInputSide, convolvedOutputSide, "Input Side vs Convoluton Side Convoluton")
    display("Testing Stereo")
    signal = np.transpose(np.array([signal, np.flip(signal)]))
    convolvedInputSide = conv.convolveInputSide(signal)
    convolvedOutputSide = conv.convolveOutputSide(signal)
    conv.plotTwoSignals(signal, convolvedInputSide, "Original vs Input Side Convoluton")
    conv.plotTwoSignals(signal, convolvedOutputSide, "Original vs Output Side Convoluton")
    conv.plotTwoSignals(convolvedInputSide, convolvedOutputSide, "Input Side vs Convoluton Side Convoluton")
    '''


