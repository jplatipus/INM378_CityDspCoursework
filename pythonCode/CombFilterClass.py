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

    #
    # constructor.
    # controlClass: the control signal calss instance used to modulate the filter
    # minDelay: minimum signal delay produced
    # maxDelay: maximum signal delay produced
    # filterSize: size of the filter to create
    # doPlot plot filter created: True yes, False no.
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
        fig = plt.figure()
        plt.title("Comb Filter Frequency Spectra")
        plt.plot(minFft, 'r', alpha=0.5, label='Min Filter')
        plt.plot(maxFft, 'b', alpha=0.5, label='Max Filter')
        plt.xlabel("Frequency")
        plt.ylabel('Amplitude')
        plt.legend(loc=4)
        fig.show()

        fig = plt.figure()
        plt.title("Comb Filter Impulse Response Plot")
        plt.plot(self.filterMin, 'r', alpha=0.5, label="Min Filter")
        plt.plot(self.filterMax, 'b', alpha=0.5, label="Max Filter")
        plt.xlabel("Impulse Sample No.")
        plt.ylabel('Amplitude')
        plt.legend(loc=4)
        fig.show()

    # Plot original signal and filtered signal
    # newSamples filtered signal
    # samples original signal
    # title title of the plot
    def plotOriginalAndConvolved(self, newSamples, samples, title):
        fig = plt.figure()
        plt.plot(samples, 'r', label='Original')
        plt.plot(newSamples, 'b', alpha=0.5, label='Filtered')
        plt.title(title)
        plt.legend(loc=4)
        plt.xlabel("Sample Number");
        plt.ylabel("Amplitude");
        fig.show()

    # Modulated filter of signal, using an interpolation of filter1 and filter2
    # using the control as the interpolation factor.
    # Convolution based on: W. Smith, Digital signal processing, Chapter 6. Input side convolution
    #
    # signal the signal to modulate
    # return filtered signal
    def filterAudioInputSide(self, signal):
        conv = self.convolutionClass.convolveInputSide(signal)
        return conv

    # Modulated filter of signal, using an interpolation of filter1 and filter2
    # using the control as the interpolation factor.
    # Convolution based on: W. Smith, Digital signal processing, Chapter 6. Output side convolution
    #
    # signal the signal to modulate
    # return filtered signal
    def filterAudioOutputSide(self, signal):
        conv = self.convolutionClass.convolveOutputSide(signal)
        return conv

    #
    # taken from lab5: plot a spectogram of the time signal (frequency over time, amplitude is the color)
    # time_signal samples to convert to frequency domain
    # samplerate: sampling rate of the signal
    # title the title to give to the plot
    def plot_spectrogram(self, time_signal, samplerate, title=''):
        frequencies, timepoints, specgram = signal.stft(time_signal, fs=samplerate, nperseg=1024)
        power_spectrogram = 20 * np.log10(np.abs(specgram) + np.finfo(float).eps)
        # adding a small number before applying log10 avoids divide by zero errors
        fig = plt.figure()
        plt.pcolormesh(timepoints, frequencies, power_spectrogram, vmin=-99, vmax=0)
        plt.title(title)
        plt.ylabel('frequency [Hz]')
        plt.xlabel('time [s]')
        #     plt.xlim(0.5e7, 0.7e7) # adjust the x-axis to zoom in on a specific time region
        #     plt.xlim(5e7, 5.5e7)
        #     plt.ylim(0, 0.0005) # adjust the y-axis range to zoom in on low frequencies
        fig.show()


TEST = True
if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    TEST = False

if TEST:

    from SoundPlayer import SoundPlayer
    # Define wav filename used:
    s1Filename = "../audio/carrier.wav"
    s2Filename = "../audio/rockA.wav"
    s3Filename = "../audio/rockB.wav"
    wavClass = WavClass(wavFileName=s1Filename, doPlots=False)
    controlClass = ControlClass(wavClass.sampleRate, wavClass.numSamples, frequencyHz=1, maxAmplitude=1.0, doPlot=True)
    filter = CombFilterClass(controlClass, minDelay=4, maxDelay=32, filterSize=64, doPlot=True)
    newSamples = filter.filterAudioInputSide(wavClass.samplesMono)
    newWavCLass = WavClass(rawSamples=newSamples, rawSampleRate=wavClass.sampleRate)
    filter.plotOriginalAndConvolved(newSamples, wavClass.samplesMono, "Original and Filtered Signal")
    filter.plotFilters()
    filter.plot_spectrogram(wavClass.samplesMono,wavClass.sampleRate, "Original Signal")
    print("Yoyo")
    filter.plot_spectrogram(newSamples, wavClass.sampleRate, "Filtered Signal")
    SoundPlayer().playWav(newSamples, wavClass.sampleRate)

