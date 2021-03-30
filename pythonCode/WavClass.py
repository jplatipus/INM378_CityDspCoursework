#
# Code for audio file loading and conversion to mono
#
import sys

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
    # isMono boolean if mono samples is 1D, False it is 2D
    # samplesMono the samples converted to mono

    def __init__(self, wavFileName=None, doPlots=False, rawSamples=[], rawSampleRate=None):
        if wavFileName != None:
            self.initFromWavFile(wavFileName, doPlots)
        elif len(rawSamples) > 0:
            self.initFromSamples(rawSamples, rawSampleRate, doPlots)
        self.numSamples = len(self.samplesMono)

    # constructor loads the given file, converts it to 1 channel mono if it is stereo
    # plots the sample, and stereo correlation information
    def initFromWavFile(self, wavFileName, doPlots):
        self.sampleRate, samples = wavfile.read(wavFileName);
        # To avoid warnings and errors of the kind:
        #   RuntimeWarning: overflow encountered in short_scalars
        # convert samples from short to float:
        self.samples = samples.astype(np.float)
        self.filename = wavFileName;
        self.doPlots = doPlots
        self.__preprocessSamples__()

    def initFromSamples(self, samples, sampleRate, doPlots):
        self.samples = samples.astype(np.float)
        self.sampleRate = sampleRate
        self.filename = "RawSamplesSUpplied.noFile"
        self.doPlots = doPlots
        self.__preprocessSamples__()

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
    # and one channel has been inverted. This function finds how many samples to shift and inverts (if needed)
    # the respective channels
    # input:
    #   flipChannel: channel to invert, None if no inversion needed
    #
    def __alignSamples__(self, flipChannel=None):
        correlationCoefficients = np.zeros(self.sampleRate)
        channel1Section = self.samples[0:self.sampleRate,0]
        channel2Section = self.samples[0:self.sampleRate,1]
        channel1RightShift = channel1Section
        maxCoefficient = 0.0
        maxCoefficientIndex = 0

        #Invert the sign of one channel?
        if flipChannel != None:
            self.samples[:,flipChannel] = self.samples[:,flipChannel] * -1
            # check correllation maybe this fixed the correlation
            rawCoeff = np.dot(self.samples[:, 0], self.samples[:, 1]) / np.sqrt(np.dot(self.samples[:, 0], self.samples[:, 0]) * np.dot(self.samples[:, 1], self.samples[:, 1]))
            if abs(rawCoeff) > 0.8:
                return

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
            channel1 = self.samples[:,0]
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

    # check DC offset of sample channels is approximately zero, if not, adjust the offset
    # if the samples member is two dimensional, try converting to mono
    # stores the conversion to mono in self.samplesMono
    def __preprocessSamples__(self):
        if (self.samples.ndim == 2):
            display("Read file: {} {} samples, rate {}".format(self.filename, len(self.samples[:, 0]), self.sampleRate))
            mean1 = np.mean(self.samples[:, 0])
            mean2 = np.mean(self.samples[:, 0])
            # adjust DC offset?
            if abs(mean1) > 0.1:
                display("Adjusting DC offset Channel0 Mean: {} Variance: {} Std Dev: {} median {}".format(np.mean(self.samples[:, 0]),
                                                                                  np.var(self.samples[:, 0]),
                                                                                  np.std(self.samples[:, 0]),
                                                                                  np.median(self.samples[:, 0])))
                self.samples[:,0] = self.samples[:,0] - mean1
            if abs(mean2) > 0.1:
                display("Adjusting DC offset Channel1 Mean: {} Variance: {} Std Dev: {}".format(np.mean(self.samples[:, 1]),
                                                                        np.var(self.samples[:, 1]),
                                                                        np.std(self.samples[:, 1]),
                                                                        np.median(self.samples[:, 0])))
                self.samples[:, 1] = self.samples[:, 1] - mean1
            self.__convertToMono__()
        else:
            display("Read file: {} {} samples, rate {}".format(self.filename, len(self.samples), self.sampleRate))
            mean = np.mean(self.samples)
            if mean > abs(0.1):
                #adjust dc offset
                display("Adjusting DC offset. Channel Mean: {} Variance: {} Std Dev: {}".format(np.mean(self.samples),
                                                                      np.var(self.samples),
                                                                      np.std(self.samples),
                                                                      np.median(self.samples)))
                self.samples = self.samples - mean
            self.samplesMono = self.samples
            self.__plotMono__("One Channel Sample File Plot")


    # Convert the stereo sample to a mono sample, displays a plot of samples and conversion
    # store the mono version in self.samplesMono
    # expects self.samples to be 2 dimensional
    def __convertToMono__(self):
        display("Convert to mono: {}".format(self.filename))
        # calculate correlation coefficient
        rawCoeff = np.dot(self.samples[:, 0], self.samples[:, 1]) / np.sqrt(np.dot(self.samples[:, 0], self.samples[:, 0]) * np.dot(self.samples[:, 1], self.samples[:, 1]))
        self.samplesMono = self.samples[:, 0] / 2 + self.samples[:, 1] / 2;
        if abs(rawCoeff) >= 0.8:
            # no need to correct the samples
            # plot them:
            self.__plotStereo__("Good conversion to mono abs(correlation coefficient) >= 0.8: {}".format(abs(rawCoeff)));
        else:
            # samples need correcting, plot the original:
            self.__plotStereo__("Poor conversion to mono abs(correlation coefficient) < 0.8, needs phase alignment: {}".format(abs(rawCoeff)))
            # correct the samples
            if (rawCoeff < 0):
                # negative value implies there is cancellation happening, invert one of the channels
                # as well as checking for correlation again, possibly shifting one channel
                display("One channel needs inverting, the correlation coefficient is < 0")
                self.__alignSamples__(flipChannel=0)
            else:
                # no inverting the sign of one channel is needed, but shifting one channel may be necessary
                self.__alignSamples()

            # now the samples have been preprocessed, convert the channels to mono into self.monoSamples
            # convert to mono
            self.samplesMono = self.samples[:, 0]/2 + self.samples[:, 1]/2;
            # plot the corrected samples
            self.__plotStereo__("Aligned conversion to mono (correlation coefficient was < 0.8): {}".format(abs(rawCoeff)))
        self.__plotMono__("Resulting Mono Sample")


# Set to True to test this class
TEST = True
if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    Test = False

if TEST:
    # Define wav filename used:
    s1Filename = "../audio/carrier.wav"
    s2Filename = "../audio/rockA.wav"
    s3Filename = "../audio/rockB.wav"
    display(s2Filename)
    wavClass = WavClass(wavFileName=s1Filename,doPlots=True)
    # code below only works in a notebook:
    #display(Audio(wavClass.samples, rate=wavClass.sampleRate, normalize=False));
    from SoundPlayer import SoundPlayer
    #SoundPlayer().playWav(wavClass.samples, wavClass.sampleRate)
