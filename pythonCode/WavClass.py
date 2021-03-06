#
# Code for audio file loading and conversion to mono
#
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

#
# Class that encapsulates the wav file functionality:
# - load a wav file
# - plot the wav file's channel(s)
# - convert to mono using correlation
# - plot results of the search for maximum correlation if the loaded stereo sample needs correcting
#
class WavClass:
    # Class properties:
    # samples the samples as read from the wav file
    # sampleRate the sample rate
    # filename the wav file name
    # isMono boolean if mono samples is 1D, Fals it is 2D
    # samplesMono the samples converted to mono

    # constructor loads the given file, converts it to 1 channel mono if it is stereo
    # plots the sample, and stereo correlation information
    def __init__(self, wavFileName, doPlots):
        self.sampleRate, self.samples = wavfile.read(wavFileName);
        self.filename = wavFileName;
        self.doPlots = doPlots
        if (self.samples.ndim == 2):
            display("Read file: {} {} samples, rate {}".format(self.filename, len(self.samples[:,0]), self.sampleRate))
            self.__convertToMono__()
        else:
            display("Read file: {} {} samples, rate {}".format(self.filename, len(self.samples[:]), self.sampleRate))
            self.samplesMono = self.samples
            self.__plotMono__("One Channel Sample File Plot")


    # Plot stereo samples, and mono samples on the same graph to see the result of
    # a conversion from stereo to mono
    # input stereo: 2d array format: [samples,channel]
    #       monoSamples: 1d array of samples
    #       title: title to give to the plot
    def __plotStereo__(self, title):
        if not self.doPlots:
            return

        fig, ax1 = plt.subplots()
        ax1.set_title(title)
        ax1.plot(self.samples[:,0], label="Ch. 1", alpha=0.35, color='green')
        ax1.plot(self.samples[:,1], label="Ch. 2", alpha=0.35, color='orange')
        ax1.plot(self.samplesMono, label="Mono", alpha=0.35, color='blue')
        ax1.legend(loc = 4)
        ax1.set_xlabel("Sample Number")
        ax1.set_ylabel("Amplitude")
        plt.show()

    # Plot monophonic (mono) samples
    #       title: title to give to the plot
    def __plotMono__(self, title):
        if not self.doPlots:
            return
        fig, ax1 = plt.subplots()
        ax1.set_title(title)
        ax1.plot(self.samplesMono, 'b')
        ax1.set_xlabel("Sample Number")
        ax1.set_ylabel("Amplitude")
        plt.show()

    # Using Audacity, on RockA.wav, it can be seen that the stereo channels are not in phase,
    # and one channel has been inverted. This function finds how many samples to shift and invert
    # the respective channels
    def __alignSamples__(self):
        correlationCoefficients = np.zeros(self.sampleRate)
        channel1Section = self.samples[0:self.sampleRate,0] * -1
        channel2Section = self.samples[0:self.sampleRate,1]
        channel1RightShift = channel1Section
        maxCoefficient = 0.0
        maxCoefficientIndex = 0

        # loop shifting channel 1 to the right to find the highest correlation coefficient between channel 1 & 2
        for offset in range(0,int(self.sampleRate/10)):
            #rawCoeff = np.dot(self.samples[:, 0], self.samples[:, 1]) / np.sqrt(np.dot(self.samples[:, 0], self.samples[:, 0]) * np.dot(self.samples[:, 1], self.samples[:, 1]))
            # calculate the correlation coefficient of the two channels at the currently shifted position
            correlationCoefficients[offset] = np.dot(channel1RightShift, channel2Section) / np.sqrt(np.dot(channel1RightShift, channel1RightShift) * np.dot(channel2Section, channel2Section))
            channel1RightShift = np.concatenate([np.zeros(1), channel1RightShift])
            channel2Section = np.concatenate([channel2Section, np.zeros(1)])
            if maxCoefficient < correlationCoefficients[offset]:
                maxCoefficient = correlationCoefficients[offset]
                maxCoefficientIndex = offset
        self.__plotChannelAlignmentCorrelationCoefficientSearch__(correlationCoefficients, maxCoefficientIndex)
        # if the correlation coefficient is > 0.8, we can correct the stereo sample
        if maxCoefficient > 0.8:
            # channel1 is inverted & prepended with maxCoefficientIndex zeros (right shifted)
            channel1 = self.samples[:,0] * -1
            channel1 = np.concatenate((np.zeros(maxCoefficientIndex), channel1), axis=None)
            # channel 2 is appended with zeros to make length same as channel1
            channel2 = self.samples[:,1]
            channel2 = np.concatenate((channel2, np.zeros(maxCoefficientIndex)), axis=None)
            # new in phase stereo is constructed
            self.samples = np.transpose(np.array([channel1, channel2]))

    # plot the results of the search for the maximum correlation coefficient
    # Input: correlationCoefficients 1d array of correlation coefficients
    #        maxCoefficientIndex index in the array of the correlation coefficient that is the largest
    def __plotChannelAlignmentCorrelationCoefficientSearch__(self, correlationCoefficients, maxCoefficientIndex):
        if not self.doPlots:
            return
        # plot all correlation coefficients calculated
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(correlationCoefficients[0:maxCoefficientIndex + 10])
        ax.set_title("correlationCoefficients")
        ax.set_xlabel("Channel 1 right shift value")
        ax.set_ylabel("Channel 1&2 correlation coefficient")

        # set xaxis labels
        ticks = np.array(range(maxCoefficientIndex + 10))
        labels = ticks.astype(str)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        plt.grid()
        plt.show()
        print("Max coefficient: ", correlationCoefficients[0:maxCoefficientIndex])

    # Convert the stereo sample to a mono sample, displays a plot of samples and conversion
    #
    def __convertToMono__(self):
        display("Converted to mono: {}".format(self.filename))
        # calculate correlation coefficient
        rawCoeff = np.dot(self.samples[:, 0], self.samples[:, 1]) / np.sqrt(np.dot(self.samples[:, 0], self.samples[:, 0]) * np.dot(self.samples[:, 1], self.samples[:, 1]))
        self.samplesMono = self.samples[:, 0] / 2 + self.samples[:, 1] / 2;
        if rawCoeff >= 0.8:
            # no need to correct the samples
            # plot them:
            self.__plotStereo__("Good conversion to mono (correlation coefficient >= 0.8:) {}".format(rawCoeff));
        else:
            # samples need correcting, plot the original:
            self.__plotStereo__("Poor conversion to mono (correlation coefficient < 0.8): {}".format(rawCoeff))
            # correct the samples
            self.__alignSamples__()
            # convert to mono
            self.samplesMono = self.samples[:, 0]/2 + self.samples[:, 1]/2;
            # plot the corrected samples
            self.__plotStereo__("Aligned conversion to mono (correlation coefficient < 0.8): {}".format(rawCoeff))
        self.__plotMono__("Resulting Mono Sample")
# Set to True to test this class
TEST = False
if TEST:
    # Define wav filename used:
    s1Filename = "../audio/carrier.wav"
    s2Filename = "../audio/rockA.wav"
    s3Filename = "../audio/rockB.wav"
    display(s2Filename)
    wavClass = WavClass(s2Filename)
    # code below only works in a notebook:
    #display(Audio(wavClass.samples, rate=wavClass.sampleRate, normalize=False));