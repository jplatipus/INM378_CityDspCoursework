from scipy.io import wavfile
from scipy import fft, signal
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# Class that applies a filter to a sample
from pythonCode.WavClass import WavClass


class FilterClass:
    # wav : WavClass
    # filteredSamples

    def __init__(self, wavClass, doPlot):
        self.wav = wavClass
        self.doPlot = doPlot

    # Filter code
    def __filterAudio__(self, samples):
        spectrum = fft.rfft(samples)
        plt.figure()
        plt.title('Frequency Spectrum of original signal')
        plt.plot(np.abs(spectrum))
        plt.show()
        num_samples = len(samples)
        freq_response = np.abs(np.cos(2 * np.pi * 0.00005 * np.arange(-num_samples / 2, 1)))

        filtered_freq = spectrum * freq_response
        if self.doPlot:
            plt.figure()
            plt.title("Frequency Filtered Spectrum")
            plt.plot(np.abs(filtered_freq), 'b', label="Signal Frequencies");
            plt.xlabel("Frequency (Hz)");
            plt.ylabel("Amplitude");
            plt.twinx()  # creates a new y-axis, because the range of y-values of spectrum and freq_response are too
            # so different that plotting them on the same graph would make freq_response invisible
            plt.plot(np.abs(freq_response), 'g', label="Filter")
            plt.xlabel("Frequency");
            plt.ylabel("Filter Value");
            plt.legend(loc=4);
            plt.show()
        self.filteredSamples = fft.irfft(filtered_freq);

    # output of convolution is the length of the signal + the length of the filter kernel - 1
    # W. Smith, Digital signal processing, Chapter 6.
    def __convolveInputSide__(self, signal, kernel):

        output = np.zeros(len(signal) + len(kernel) - 1)

        for signalIndex in range(len(signal)):
            for kernelIndex in range(len(kernel)):
                output[signalIndex + kernelIndex] = output[signalIndex + kernelIndex] + signal[signalIndex] * kernel[kernelIndex]
        return output

    # output of convolution is the length of the signal + the length of the filter kernel - 1
    # W. Smith, Digital signal processing, Chapter 6.
    def __convolveOutputSide__(selfself, signal, kernel):
        output = np.zeros(len(signal) + len(kernel) - 1)
        output2 = np.ones(len(signal) + len(kernel) - 1)
        outputRange = range(len(output))
        kernelRange = range(len(kernel))
        for outputIndex in outputRange:
            output2[outputIndex] = 0.0
            for kernelIndex in kernelRange:
                if (outputIndex - kernelIndex < 0) or (outputIndex - kernelIndex > (len(signal) - 1)):
                    # don't do anything
                    pass
                else:
                    output[outputIndex] = output[outputIndex] + kernel[kernelIndex] * signal[outputIndex - kernelIndex]
                    output2[outputIndex] = output2[outputIndex] + kernel[kernelIndex] * signal[outputIndex - kernelIndex]
        return output, output2

    # Perform convolution
    # doMine performs it using the convolution coded here if true, uses python's convolution if false
    def filterSignal(self, samples, filterMin, filterMax, doMine):
        newSamples = np.zeros(1)
        if doMine:
            newSamples = self.__convolveInputSide__(samples, np.concatenate([filterMin, filterMax])) / 2
        else:
            newSamples = signal.convolve(samples, np.concatenate([filterMin, filterMax]), mode='same') / 2
        self.__plotOriginalAndConvolved__(doMine, newSamples, samples)
        return newSamples

    def __filterAudioStepped__(self, samples, filterMin, filterMax, doMine):
        windowSteps = int(len(samples) / len(filterMin))
        stepSize = len(filterMin)
        newSamples = np.zeros(1)
        filter = filterMin
        for step in range(windowSteps):
            sampleWindowStart = step * stepSize
            sampleWindow = samples[sampleWindowStart:sampleWindowStart + stepSize]
            if doMine:
                conv_samples = self.__convolveInputSide__(sampleWindow, np.concatenate([filter, filter])) / 2
            else:
                conv_samples = signal.convolve(sampleWindow, np.concatenate([filter, filter]), mode='same') / 2
            newSamples = np.concatenate([newSamples, conv_samples])
            if step % 2 > 0:
                filter = filterMax
            else:
                filter = filterMin
        self.__plotOriginalAndConvolved__(doMine, newSamples, samples)
        return newSamples

    def __plotOriginalAndConvolved__(self, doMine, newSamples, samples):
        plt.figure()
        plt.plot(samples, 'r', label='Original')
        plt.plot(newSamples, 'b', alpha=0.5, label='Filtered')
        plt.title("Filtered My Convolution: {}".format(doMine))
        plt.legend(loc=4)
        plt.xlabel("Sample Number");
        plt.ylabel("Amplitude");
        plt.show()

    '''
        frequencyResponseSlow = np.abs(np.sin(2*np.pi*0.0002*np.arange(-5000, 1)))
        plt.plot(frequencyResponseSlow, 'b');
        slowFilterCoefficient = fft.irfft(frequencyResponseSlow);
        frequencyResponseFast = np.abs(np.sin(2*np.pi*0.0055*np.arange(-5000, 1)))
        fastFilterCoefficient = fft.irfft(frequencyResponseFast);
        plt.plot(np.abs(frequencyResponseFast), 'r');
        plt.show()
        #plt.plot(slowFilterCoefficient[5000:195000], 'b');
        plt.plot(np.abs(slowFilterCoefficient), 'b');
        plt.show()
        #plt.plot(slowFilterCoefficient[5000:195000], 'b');
        plt.plot(np.abs(slowFilterCoefficient), 'b');
        #plt.show()
        #plt.plot(fastFilterCoefficient[5000:195000], 'r', alpha=0.5);
        plt.plot(np.abs(fastFilterCoefficient), 'r', alpha=0.5);
        plt.show()
        return slowFilterCoefficient, fastFilterCoefficient
    '''

    def createCombFilters(self, filterSize, minDelay, maxDelay):
        # impulse signal for filter1
        filterMin = np.zeros(filterSize)
        filterMin[0] = 1
        filterMin[minDelay] = 0.75
        # impulse signal for filter 2
        filterMax = np.zeros(filterSize)
        filterMax[0] = 1
        filterMax[maxDelay] = 0.75

        # plot the filters' frequency spectrum
        minFft = fft.rfft(filterMin)
        maxFft = fft.rfft(filterMax)
        if self.doPlot:
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

        return filterMin, filterMax

    def filterAudio(self):
        self.__filterAudio__(self.wav.samplesMono)

    def filterAudioStepped(self, filterMin, filterMax, doManualConvolution):
        self.__filterAudioStepped__(self.wav.samplesMono, filterMin, filterMax, doManualConvolution)

Test = True
# TEST of a simple sine signal convolved using input side, output side and using python's implementation
# All 3 are plotted on top of each other to check that they produce the same result.
#
if Test:
    # Define wav filename used:
    s1Filename = "../audio/carrier.wav"
    s2Filename = "../audio/rockA.wav"
    s3Filename = "../audio/rockB.wav"
    wavClass = WavClass()
    sig = np.abs(np.sin(2 * np.pi * 0.05 * np.arange(-80, 1)))
    plt.figure()
    plt.plot(sig)
    plt.title("Signal for test")
    plt.show()
    kernel = np.zeros(30)
    kernel[0] = 1.0
    kernel[15] = 0.75
    filterClass = FilterClass(wavClass, True)
    inputSide = filterClass.__convolveInputSide__(sig, kernel)
    outputSide, outputSide2 = filterClass.__convolveOutputSide__(sig, kernel)
    conv = signal.convolve(sig, kernel)
    plt.figure()
    plt.plot(inputSide, 'r', label='iSide')
    plt.plot(outputSide, 'b', alpha = 0.5, label='oSide')
    plt.plot(conv, 'g', alpha=0.5, label='oSide2')
    plt.title("PLot of 3 convolution operations: input side, output side and Python's")
    plt.legend()
    plt.show()

Test = False
# TEST creating the convolution comb filter min max max values which need to be used in the
# interpolation. Plot the impulse response as well as the frequency spectrum of the filter.
if Test:
    # Define wav filename used:
    s1Filename = "../audio/carrier.wav"
    s2Filename = "../audio/rockA.wav"
    s3Filename = "../audio/rockB.wav"
    display(s1Filename)
    wavClass = WavClass(wavFileName=s2Filename, doPlots=False)
    filterClass = FilterClass(wavClass, True)
    #filterClass.filterAudio()
    filterMin, filterMax = filterClass.createCombFilters(64, 8, 32)
    newSamples = filterClass.filterSignal(wavClass.samplesMono, filterMin, filterMax, True)
    newSample2 = filterClass.filterSignal(wavClass.samplesMono, filterMin, filterMax, False)
    #display(Audio(newSamples, rate=s3Rate))