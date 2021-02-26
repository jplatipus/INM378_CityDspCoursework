import numpy as np
from scipy import fft, signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.util import random_noise
from IPython.display import Audio, display

# Convert the stereo sample to a mono sample
# input: sampled2D a 2d array of sample values
#        sampleRate the smpling rate of the samples
# returns a 1d array of the samples converted to mono
def convertToMono(samples2D, sampleRate):
    display("Convert to mono")
    monoSamples = (samples2D[:,0] + samples2D[1]) / 2;
    return monoSamples

def loadAudioAsMono(filename):
    sampleRate, samples = wavfile.read(filename);
    # if stereo file, convert to mono
    if (samples.ndim == 2):
        samples = convertToMono(samples, sampleRate)
    return samples, sampleRate

# write your code here
s1, s1Rate = loadAudioAsMono("audio/carrier.wav")
s2, s2Rate = loadAudioAsMono("audio/rockA.wav")
s3, s3Rate = loadAudioAsMono("audio/rockB.wav")
display(Audio(s1, rate=s1Rate));
display(Audio(s2, rate=s2Rate));
display(Audio(s3, rate=s3Rate));
Audio(s1, rate=s1Rate)