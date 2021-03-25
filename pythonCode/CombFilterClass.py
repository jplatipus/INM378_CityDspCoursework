import sys

from scipy import fft, signal
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from pythonCode.ControlClass import ControlClass
from pythonCode.ConvolutionClass import ConvolutionClass
from pythonCode.WavClass import WavClass

#
# Modulated Comb filter class (flanger): modulated between 2 filters whose
# delay is above 20 Hz in order to create a flanging effect
#
class CombFilterClass:

    def __init__(self, controlClass, minDelay=2, maxDelay=1022, filterSize=1024, doPlot=False):
        self.doPlot = doPlot
        # impulse signal for filter1
        filterMin = np.zeros(filterSize)
        filterMin[0] = 1
        filterMin[minDelay] = 0.75
        # impulse signal for filter 2
        filterMax = np.zeros(filterSize)
        filterMax[0] = 1
        filterMax[maxDelay] = 0.75
        self.filterMin = filterMin
        self.filterMax = filterMax
        if (self.doPlot):
            self.plotFilters()

        self.convolutionClass = ConvolutionClass(filterMin, filterMax, controlClass)

    # plot the given filters, overlayed
    def plotFilters(self):
        # plot the filters' frequency spectrum
        minFft = fft.rfft(self.filterMin)
        maxFft = fft.rfft(self.filterMax)
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
        plt.plot(self.filterMin, 'r', alpha=0.5, label="Min Filter")
        plt.plot(self.filterMax, 'b', alpha=0.5, label="Max Filter")
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

    # Modulated filter of signal, using an interpolation of filter1 and filter2
    # using the control as the interpolation factor.
    # Convolution based on: W. Smith, Digital signal processing, Chapter 6. Input side convolution
    #
    # signal the signal to modulate
    # filter1 & filter 2 the filters to interpolate for each signal sample
    # control the control used in the interpolation of the two filters
    # return filtered signal
    def filterAudioInputSide(self, signal):
        conv = self.convolutionClass.convolveInputSide(signal)
        return conv

    def rescale(self, conv):
        min = np.min(conv)
        max = np.max(conv)
        if conv.ndim == 2:
            scaled = np.zeros(shape=(len(conv[:,0]), 2))
            scaled[:,0] = np.interp(conv[:, 0], (min, max), (-0.8, +0.8))
            scaled[:,1] = np.interp(conv[:, 1], (min, max), (-0.8, +0.8))
        else:
            scaled = np.interp(conv, (min, max), (-0.8, 0.8))
        return scaled

    def filterAudioOutputSide(self, signal):
        conv = self.convolutionClass.convolveOutputSide(signal)
        return conv

TEST = True
if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    TEST = False

if TEST:

    from SoundPlayer import SoundPlayer
    # Define wav filename used:
    s1Filename = "../audio/carrier.wav"
    s2Filename = "../audio/rockA.wav"
    s3Filename = "../audio/rockB.wav"
    wavClass = WavClass(wavFileName=s2Filename, doPlots=False)
    controlClass = ControlClass(wavClass.sampleRate, wavClass.numSamples, frequencyHz=1, maxAmplitude=1.0, doPlot=True)
    filter = CombFilterClass(controlClass, minDelay=4, maxDelay=32, filterSize=64, doPlot=True)
    newSamples = filter.filterAudioInputSide(wavClass.samplesMono)
    newWavCLass = WavClass(rawSamples=newSamples, rawSampleRate=wavClass.sampleRate)
    filter.plotOriginalAndConvolved(newSamples, wavClass.samplesMono, "Original and Filtered Signal")
    SoundPlayer().playWav(newSamples, wavClass.sampleRate)

