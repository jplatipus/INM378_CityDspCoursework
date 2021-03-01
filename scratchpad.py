import numpy as np
import numpy as np
from scipy import fft, signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.util import random_noise
from IPython.display import Audio, display

s1Filename = "audio/carrier.wav"
s2Filename = "audio/rockA.wav"
s3Filename = "audio/rockB.wav"


#
# Code for audio file loading and conversion to mono
#

# Plot stereo samples, and mono samples on the same graph to see the result of
# a conversion from stereo to mono
# input stereo: 2d array format: [samples,channel]
#       monoSamples: 1d array of samples
#       title: title to give to the plot
def plotStereo(stereo, monoSamples, title):
    fig, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.plot(stereo[:,0], label="Ch. 1", alpha=0.35, color='green')
    ax1.plot(stereo[:,1], label="Ch. 2", alpha=0.35, color='orange')
    ax1.plot(monoSamples, label="Mono", alpha=0.35, color='blue')
    ax1.legend(loc = 4)
    ax1.set_xlabel("Sample Number")
    ax1.set_ylabel("Amplitude")
    plt.show()

# Plot monophonic (mono) samples
#       monoSamples: 1d array of samples
#       title: title to give to the plot
def plotMono(monoSamples, title):
    fig, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.plot(monoSamples, 'b')
    ax1.set_xlabel("Sample Number")
    ax1.set_ylabel("Amplitude")
    plt.show()

# Using Audacity, on RockA.wav, it can be seen that the stereo channels are not in phase,
# and one channel has been inverted. This function finds how many samples to shift and invert
# the respective channels
# Input: stereo 2d list of channel1 and channel2 samples
#        sampleRate: sampling rate
# Output: unchanged stereo if no good correlation found, corrected samples otherwise.
def alignSamples(stereo, sampleRate):
    correlationCoefficients1 = np.zeros(sampleRate)
    correlationCoefficients = np.zeros(sampleRate)
    channel1Section = stereo[0:sampleRate,0] * -1
    channel2Section = stereo[0:sampleRate,1]
    channel1RightShift = channel1Section
    maxCoefficient = 0.0
    maxCoefficientIndex = 0

    for offset in range(0,int(sampleRate/10)):
        rawCoeff = np.dot(stereo[:, 0], stereo[:, 1]) / np.sqrt(np.dot(stereo[:, 0], stereo[:, 0]) * np.dot(stereo[:, 1], stereo[:, 1]))
        correlationCoefficients[offset] = np.dot(channel1RightShift, channel2Section) / np.sqrt(np.dot(channel1RightShift, channel1RightShift) * np.dot(channel2Section, channel2Section))
        channel1RightShift = np.concatenate([np.zeros(1), channel1RightShift])
        channel2Section = np.concatenate([channel2Section, np.zeros(1)])
        if maxCoefficient < correlationCoefficients[offset]:
            maxCoefficient = correlationCoefficients[offset]
            maxCoefficientIndex = offset
    # plot all correlation coefficients calculated
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(correlationCoefficients[0:maxCoefficientIndex+10])
    ax.set_title("correlationCoefficients")
    ax.set_xlabel("Channel 1 right shift value")
    ax.set_ylabel("Channel 1&2 correlation coefficient")
    # set xaxis labels
    ticks = np.array(range(maxCoefficientIndex+10))
    labels = ticks.astype(str)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize='small')
    plt.show()
    print("Max coefficient: ", maxCoefficient)
    # if the correlation coefficient is > 0.8, we can correct the stereo sample
    if maxCoefficient > 0.8:
        # channel1 is inverted & prepended with maxCoefficientIndex zeros (right shifted)
        channel1 = stereo[:,0] * -1
        channel1 = np.concatenate((np.zeros(maxCoefficientIndex), channel1), axis=None)
        # channel 2 is appended with zeros to make length same as channel1
        channel2 = stereo[:,1]
        channel2 = np.concatenate((channel2, np.zeros(maxCoefficientIndex)), axis=None)
        # new in phase stereo is constructed
        stereo = np.transpose(np.array([channel1, channel2]))
    return stereo

# Convert the stereo sample to a mono sample, displays a plot of samples and conversion
# input: sampled2D a 2d array of sample values
#        sampleRate the smpling rate of the samples
# returns a 1d array of the samples converted to mono
def convertToMono(stereo, sampleRate, filename):
    display("Converted to mono: {}".format(filename))
    # calculate correlation coefficient
    rawCoeff = np.dot(stereo[:, 0], stereo[:, 1]) / np.sqrt(np.dot(stereo[:, 0], stereo[:, 0]) * np.dot(stereo[:, 1], stereo[:, 1]))
    monoSamples = stereo[:, 0] / 2 + stereo[:, 1] / 2;
    if rawCoeff >= 0.8:
        # no need to correct the samples
        # plot them:
        plotStereo(stereo, monoSamples, "Good conversion to mono (correlation coefficient >= 0.8:) {}".format(rawCoeff));
    else:
        # samples need correcting, plot the original:
        plotStereo(stereo, monoSamples, "Poor conversion to mono (correlation coefficient < 0.8): {}".format(rawCoeff))
        # correct the samples
        stereo = alignSamples(stereo, sampleRate)
        # convert to mono
        monoSamples = stereo[:, 0]/2 + stereo[:, 1]/2;
        # plot the corrected samples
        plotStereo(stereo, monoSamples, "ALigned conversion to mono (correlation coefficient < 0.8): {}".format(rawCoeff))
    return monoSamples

# Load a given audio wav file and convert to mono (if it is stereo, also plotted).
def loadAudioAsMono(filename):
    sampleRate, samples = wavfile.read(filename);
    # if stereo file, convert to mono
    if (samples.ndim == 2):
        samples = convertToMono(samples, sampleRate, filename)
    else:
        plotMono(samples, sampleRate)
    return samples, sampleRate

display(s2Filename)
s2, s2Rate = loadAudioAsMono(s2Filename)
#display(Audio(s2, rate=s2Rate));







ar1 = [0.1, 0.2, 1.1, 1.2]
ar2 = [0.3, 0.4, 1.3, 1.4]
crap = np.array([ar1, ar2])
print(crap)
print(type(crap))
arr2D = np.array([[19, 21], [11, 46]])
print(arr2D)
print(type(arr2D))

npar = np.array([1, 2, 3, 4, 5])
tickn = np.array(range(len(npar)))
ticks = tickn.astype(str)
print(tickn, ticks)
