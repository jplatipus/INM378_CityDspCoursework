import sys

from scipy import fft, signal
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from pythonCode.WavClass import WavClass

#
# Modulated Comb filter class (flanger): modulated between 2 filters whose
# delay is above 20 Hz in order to create a flanging effect
#
class CombFilterClass:

    def __init__(self):
        self.doPlot = True

    # wav : WavClass
    # filteredSamples
    def __initSignalWav__(self, wavClass, doPlot):
        self.wav = wavClass
        self.doPlot = doPlot

    # generate a repeating sine control:
    # size of control signal might as well be the same as the sample size
    # Input:
    #  sampleRate: sample rate of signal to produce (44100 is common)
    #  numSamples: the size of the array of sines to return
    #  frequency: the frequency of the sine wave(s) wanted
    #  amplitude: the maximum value of the sine
    # Return:
    #  array of repeating sines
    def createControlSignal(self, sampleRate=800, numSamples=1600, frequencyHz=2, amplitude=1.0):
        # number points for 2hz sine for given sample rate:
        numPoints = sampleRate / frequencyHz # eg: 2 hertz = 2 samples per second
        # points needed to represent the sine control signal:
        timepoints = np.arange(0, 1, 1/numPoints)
        # convert the 0-1 values to sine(0-1):
        controlSignal = np.sin(2 * np.pi * frequencyHz * timepoints )
        # apply amplitude factor to sines:
        controlSignal = controlSignal * amplitude
        # repeat the sines to build an array at least as long as numSamples
        repeats = int(numSamples / len(controlSignal)) + 1
        controlSignal = np.tile(controlSignal, repeats)
        # trim off extra values
        controlSignal = controlSignal[:numSamples]
        # plot control signal:
        if self.doPlot:
            self.plotControlSignal(controlSignal)
        # return control signal:
        return controlSignal

    # plot the control signal
    def plotControlSignal(self, controlSignal):
        plt.figure()
        plt.plot(controlSignal)
        plt.title("Comb Filter Control Signal (sine 2Hz)")
        plt.xlabel("Sample Number")
        plt.ylabel('Amplitude')
        plt.show()

    # create 2 comb (delay) filters
    # minDelay: delay in number of samples for first filter
    # maxDelay: delay in samples for second filter
    # filterSize: filter size in samples (size must be >= minDelay & maxDelay)
    # return minDelay filter, maxDelayFilter
    def CreateCombFilters(self, minDelay=4, maxDelay=32, filterSize=64):
        # impulse signal for filter1
        filterMin = np.zeros(filterSize)
        filterMin[0] = 1
        filterMin[minDelay] = 0.95
        # impulse signal for filter 2
        filterMax = np.zeros(filterSize)
        filterMax[0] = 1
        filterMax[maxDelay] = 0.95
        if (self.doPlot):
            self.plotFilters(filterMin, filterMax)
        return filterMin, filterMax

    # plot the given filters, overlayed
    def plotFilters(self, filterMin, filterMax):
        # plot the filters' frequency spectrum
        minFft = fft.rfft(filterMin)
        maxFft = fft.rfft(filterMax)
        plt.figure()
        plt.title("Comb Filter Frequency Spectra")
        plt.plot(minFft, 'r', alpha=0.5, label='Min Filter')
        plt.plot(maxFft, 'b', alpha=0.5, label='Max Filter')
        plt.xlabel("Frequency")
        plt.ylabel('Amplitude')
        plt.legend(loc=4)
        plt.show()

        plt.figure()
        plt.title("Comb Filter Impulse Response Plot")
        plt.plot(filterMin, 'r', alpha=0.5, label="Min Filter")
        plt.plot(filterMax, 'b', alpha=0.5, label="Max Filter")
        plt.xlabel("Impulse Sample No.")
        plt.ylabel('Amplitude')
        plt.legend(loc=4)
        plt.show()

    # Plot original signal and filtered signal
    # newSamples filtered signal
    # samples original signal
    # title title of the plot
    def plotOriginalAndConvolved(self, newSamples, samples, title):
        plt.figure()
        plt.plot(samples, 'r', label='Original')
        plt.plot(newSamples, 'b', alpha=0.5, label='Filtered')
        plt.title(title)
        plt.legend(loc=4)
        plt.xlabel("Sample Number");
        plt.ylabel("Amplitude");
        plt.show()

    # interpolate 2 filters using the given factor
    # return a the interpolated filter
    def interpolateFilters(self, filter1, filter2, factor):
        interpolated1 = filter1 * factor
        interpolated2 = filter2 * (1 - factor)
        return interpolated1 + interpolated2

    # Modulated filter of signal, using an interpolation of filter1 and filter2
    # using the control as the interpolation factor.
    # Convolution based on: W. Smith, Digital signal processing, Chapter 6. Input side convolution
    #
    # signal the signal to modulate
    # filter1 & filter 2 the filters to interpolate for each signal sample
    # control the control used in the interpolation of the two filters
    # return filtered signal
    def filterAudio(self, signal, filter1, filter2, control):
        # start filter is filter 1, no interpolation
        filter = filter1
        # initialize convolved result to zeros: length is length of signal + padding for convolution (length of filter - 1)
        output = np.zeros(len(signal) + len(filter1) - 1)
        # for each signal value
        for signalIndex in range(len(signal)):
            # get a new interpolation of the filters
            filter = self.interpolateFilters(filter1, filter2, control[signalIndex])
            # convolve the current signal value with the filter, into the convolved result
            for filterIndex in range(len(filter)):
                output[signalIndex + filterIndex] = output[signalIndex + filterIndex] + signal[signalIndex] * filter[filterIndex]
        # return the convolution result, with the padding removed.
        return output[:-len(filter) - 1]

TEST = True
if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    TEST = False

if TEST:
    from SoundPlayer import SoundPlayer
    # Define wav filename used:
    s1Filename = "../audio/carrier.wav"
    s2Filename = "../audio/rockA.wav"
    s3Filename = "../audio/rockB.wav"
    display(s1Filename)
    wavClass = WavClass(wavFileName=s1Filename, doPlots=False)
    filter = CombFilterClass()
    controlSignal = filter.createControlSignal(wavClass.sampleRate, len(wavClass.samplesMono))
    filterMin, filterMax = filter.CreateCombFilters()
    newSamples = filter.filterAudio(wavClass.samplesMono, filterMin, filterMax, controlSignal)
    newWavCLass = WavClass(rawSamples=newSamples, rawSampleRate=wavClass.sampleRate)
    filter.plotOriginalAndConvolved(newSamples, wavClass.samplesMono, "Original and Filtered Signal")
    SoundPlayer().playWav(newSamples, wavClass.sampleRate)

