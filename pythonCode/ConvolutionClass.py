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
#
# The class encapsulates the functionality required to implement the convolution of an input signal.
# The class uses a control signal to modulate the filter kernel used to generate the convolved output signal.
#
    # Constructor:
    # filter1 and filter 2 are the two two static filters (arrays) that the control signal interpolates between
    # control is the control signal used to modulate the output of the convolution
    def __init__(self, filter1, filter2, control):
        self.filter1 = filter1
        self.filter2 = filter2
        self.control = control

    # convolve the input signal using input side convolution
    # output of convolution is the length of the signal + the length of the filter kernel - 1
    # W. Smith, Digital signal processing, Chapter 6.
    def convolveInputSide(self, signal):
        isStereo = False
        filterLength = len(self.filter1)
        # initialise output array, get length of signal. NOte the length of the array is the length of the signal + the
        # filter length, to allow for convolution's operation which generates extra values.
        # Once the convolution has completed, the extra values at the end of the output are trunctated
        # both of these operations differ between mono and stereo (1d and 2d array access)
        if signal.ndim == 2:
            # stereo
            signalLength = len(signal[:,0])
            output = np.zeros(signalLength + filterLength - 1)
            isStereo = True
            output = np.transpose(np.array([output, output]))
        else:
            # mono
            signalLength = len(signal)
            output = np.zeros(signalLength + filterLength - 1)

        # for each sample in the signal
        for signalIndex in range(signalLength):
            for kernelIndex in range(filterLength):
                # create interpolated filter using the control signal as a factor for interpolation
                filter = self.control.interpolateFilters(signalIndex, self.filter1, self.filter2)
                # convolve the sample and store it in the output
                if isStereo:
                    output[signalIndex + kernelIndex, 0] = output[signalIndex + kernelIndex, 0] + signal[signalIndex, 0] * \
                                                        filter[kernelIndex]
                    output[signalIndex + kernelIndex, 1] = output[signalIndex + kernelIndex, 1] + signal[signalIndex, 1] * \
                                                        filter[kernelIndex]
                else:
                    output[signalIndex + kernelIndex] = output[signalIndex + kernelIndex] + signal[signalIndex] * \
                                                        filter[kernelIndex]
        # return the truncated output (remove padding added for the filter length at the start)
        if isStereo:
            return output[:-(len(filter) - 1),:]
        else:
            return output[:-(len(filter) - 1)]

    # Convolve the input signal using output side convolution
    # output of convolution is the length of the signal + the length of the filter kernel - 1
    # W. Smith, Digital signal processing, Chapter 6.
    def convolveOutputSide(self, signal):
        isStereo = False
        filterLength = len(self.filter1)
        # initialise output array, get length of signal. NOte the length of the array is the length of the signal + the
        # filter length, to allow for convolution's operation which generates extra values.
        # Once the convolution has completed, the extra values at the end of the output are trunctated
        # both of these operations differ between mono and stereo (1d and 2d array access)
        if signal.ndim == 2:
            # stereo
            signalLength = len(signal[:, 0])
            output = np.zeros(signalLength + filterLength - 1)
            isStereo = True
            output = np.transpose(np.array([output, output]))

        else:
            # mono
            signalLength = len(signal)
            output = np.zeros(signalLength + filterLength - 1)

        # the array to iterate over is the output array, with space padded with zero's (the length of the filter).
        outputRange = signalLength + filterLength - 1
        kernelRange = len(self.filter1)
        # for each output sample
        for outputIndex in range(outputRange):
            # calculate the value of the output sample
            for kernelIndex in range(kernelRange):
                filter = self.control.interpolateFilters(outputIndex, self.filter1, self.filter2)
                # is this outside the output array's bounds? If so do not try to set it
                if (outputIndex - kernelIndex < 0) or (outputIndex - kernelIndex > (len(signal) - 1)):
                    # don't do anything
                    pass
                else:
                    # multiply the input sample by the impulse response
                    if isStereo:
                        output[outputIndex, 0] = output[outputIndex, 0] + filter[kernelIndex] * signal[
                            outputIndex - kernelIndex, 0]
                        output[outputIndex, 1] = output[outputIndex, 1] + filter[kernelIndex] * signal[
                            outputIndex - kernelIndex, 1]
                    else:
                        output[outputIndex] = output[outputIndex] + filter[kernelIndex] * signal[outputIndex - kernelIndex]
        # return the truncated output (remove padding added for the filter length at the start)
        if isStereo:
            return output[:-(len(self.filter1) - 1), :]
        else:
            return output[:-(len(self.filter1) - 1)]

    # Plot signal 1 (original signal) and signal 2 (filtered version) 1d (mono) arrays
    # expects 1d signal
    def plotSignals(self, signal1, signal2, title):
        plt.figure()
        plt.plot(signal1, 'r', label='Original')
        plt.plot(signal2, 'b', alpha=0.5, label='Filtered')
        plt.title(title)
        plt.legend(loc=4)
        plt.xlabel("Sample Number");
        plt.ylabel("Amplitude");
        plt.show()

    #
    # plot the two given signals (stereo) over each other, one plot per channel
    #
    def plotTwoSignals(self, signal1, signal2, title):
        if signal1.ndim == 2:
            title1 = "{} Channel 1".format(title)
            self.plotSignals(signal1[:,0], signal2[:,0], title1)
            title2 = "{} Channel 2".format(title)
            self.plotSignals(signal1[:, 1], signal2[:, 1], title2)
        else:
            self.plotSignals(signal1, signal2, title)

# TEST creating the convolution comb filter min max max values which need to be used in the
# interpolation. Plot the impulse response as well as the frequency spectrum of the filter.
Test = True

if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    Test = False
if Test:
    sig = np.abs(np.sin(2 * np.pi * 0.05 * np.arange(-80, 1)))
    plt.figure()
    plt.plot(sig)
    plt.title("Signal for test")
    plt.show()
    #control = ControlClass()
    filter = np.zeros(64)
    filter[0] = 1
    # in this test there is no convolution
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
    conv.plotSignals(sig, inputSide, "Input Side Convolution")
    conv.plotSignals(sig, inputSide, "Output Side Convolution")
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


