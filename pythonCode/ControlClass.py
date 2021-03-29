import sys

import matplotlib.pyplot as plt
import numpy as np

class ControlClass:

    # generate a repeating sine control:
    # size of control signal might as well be the same as the sample size
    # Input:
    #  sampleRate: sample rate of signal to produce (44100 is common)
    #  numSamples: the size of the array of sines to return
    #  frequency: the frequency of the sine wave(s) wanted
    #  amplitude: the maximum value of the sine
    # Sets member:
    #  self.controlSignal: array of repeating sines
    def __init__(self, sampleRate, numSamples, frequencyHz=2, maxAmplitude=1.0, doPlot=False):
        self.sampleRate = sampleRate
        self.numSamples = numSamples
        self.frequency = frequencyHz
        self.maxAmplitude = maxAmplitude

        # number points for 2hz sine for given sample rate:
        numPoints = sampleRate / frequencyHz  # eg: 2 hertz = 2 samples per second
        # points needed to represent the sine control signal:
        timepoints = np.arange(0, 1, 1 / numPoints)
        # convert the 0-1 values to sine(0-1):
        controlSignal = np.sin(2 * np.pi * frequencyHz * timepoints)
        # apply amplitude factor to sines:
        controlSignal = controlSignal * maxAmplitude
        # repeat the sines to build an array at least as long as numSamples
        repeats = int(numSamples / len(controlSignal)) + 1
        controlSignal = np.tile(controlSignal, repeats)
        # trim off extra values
        controlSignal = controlSignal[:numSamples]
        # plot control signal:
        self.controlSignal = controlSignal
        if doPlot:
            self.plotControlSignal()
        # return control signal:


    # plot the control signal
    def plotControlSignal(self):
        plt.figure()
        plt.plot(self.controlSignal)
        plt.title("Control Signal (sine {}Hz) s.rate {}, len. {}".format(self.frequency, self.sampleRate, self.numSamples))
        plt.xlabel("Sample Number")
        plt.ylabel('Amplitude')
        plt.ylim(top=1)
        plt.show()

    def getControlFactor(self, controlIndex):
        # when convolution is performed, the index goes beyond the
        # length of the input signal by definition of convolution
        # this method reflects back in this case
        if (controlIndex >= self.numSamples):
            controlIndex = (self.numSamples % controlIndex) - self.numSamples
        return self.controlSignal[controlIndex % self.numSamples]

    # interpolate 2 filters using the given controlFactor (between 0 and 1)
    # return a the interpolated filter
    def interpolateFilters(self, controlIndex, filterMin, filterMax):
        controlFactor = self.getControlFactor(controlIndex)
        interpolated1 = filterMin * controlFactor
        interpolated2 = filterMax * (1 - controlFactor)
        return interpolated1 + interpolated2


# TEST creating the convolution comb filter min max max values which need to be used in the
# interpolation. Plot the impulse response as well as the frequency spectrum of the filter.
Test = False

if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    Test = False
if Test:
    control = ControlClass(sampleRate=800, numSamples=1600, maxAmplitude=0.75, doPlot=True)
    control.plotControlSignal()
    filterMin = np.zeros(64)
    filterMin[0] = 1
    filterMin[4] = 0.95
    filterMax = np.zeros(64)
    filterMin[0] = 1
    filterMin[48] = 0.95
    print("Interpolated ", control.interpolateFilters(32, filterMin, filterMax))