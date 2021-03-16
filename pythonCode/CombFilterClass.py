from scipy import fft, signal
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from pythonCode.WavClass import WavClass

'''
create original_signal, s
create sine wave control signal, c
create two filters (pass bands in my case), b1 and b2
loop though s and at regular intervals do:
·   get a single control value, 'cv',  from sine wave
·   create new filter coefficients b_comb by interpolating between b1 and b2, using cv as interpolation factor
·   convolve the s (in the appropriate range) with b_comb
·   append to output signal
'''
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
    def createControlSignal(self, sampleRate=800, numSamples=1600, frequencyHz=2, amplitude=0.8):
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
            plt.figure()
            plt.plot(controlSignal)
            plt.title("Comb Filter Control Signal (sine 2Hz)")
            plt.xlabel("Sample Number")
            plt.ylabel('Amplitude')
            plt.show()
        # return control signal:
        return controlSignal

TEST = True
filter = CombFilterClass()
filter.createControlSignal(800, 1600)