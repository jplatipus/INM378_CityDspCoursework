from pythonCode.FinancialDataClass import FinancialDataClass
import sys
import matplotlib.pyplot as plt
from scipy import fft, signal
import numpy as np
import math
from dsp_ap.operations import circ_convolve

#
#
class FinancialFft:

    def __init__(self, financialClassInstance):
        self.financial = financialClassInstance

    def plotAllColumns(self, title):
        dataColumnIndeces = self.financial.getColumnIndeces()
        columns = 3
        rows = int(len(dataColumnIndeces) / 3 + int(len(dataColumnIndeces) % 3))
        fig = plt.figure("MainFigure")
        fig.subplots_adjust(hspace=0.6)
        fig.suptitle(title)

        for dataColumnIndex in range(0, len(dataColumnIndeces)):
            ax = fig.add_subplot(rows, columns, dataColumnIndex + 1)
            columnName = self.financial.getColumnName(dataColumnIndex)
            ax.title.set_text(columnName)
            self.plotColumn(columnName, ax)
        plt.show()

    # make sure signal length is power of 2, pad with zero's if not to
    # make the length a power of 2
    def padToPower2Length(self, data):
        length = len(data)
        while True:
            root = int(math.sqrt(length))
            if root ** 2 == length:
                break
            data = np.append(data, 0.0)
            length = length + 1
        return data


    def plotColumn(self, columnName, axis=None, hanWindowSize=None):
        if axis == None:
            fig = plt.figure()
            ax = fig.subplots()
        else:
            ax = axis
        columnData = self.financial.getExpWeightedColumnData(columnName)
        columnData = self.padToPower2Length(columnData)
        signalData = columnData
        if hanWindowSize != None:
            window = signal.hann(hanWindowSize)
            #signalData = signal.convolve(window, signalData, mode="same")
            signalData = circ_convolve(window, signalData)
        spectrum = fft.rfft(signalData)
        ax.plot(np.abs(np.log10(spectrum)))
        if axis == None:
            ax.title.set_text("Frequency Spectrum of {}".format(columnName))
            totalMonths = self.financial.getTotalMonthsInData()
            plt.xlabel('frequency [months]')
            plt.ylabel('amplitude')
            plt.show()
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
        plt.xlabel('time [Years]')
        #     plt.xlim(0.5e7, 0.7e7) # adjust the x-axis to zoom in on a specific time region
        #     plt.xlim(5e7, 5.5e7)
        #     plt.ylim(0, 0.0005) # adjust the y-axis range to zoom in on low frequencies
        fig.show()

    def plotColumnSpectrum(self, columnName):
        columnData = self.financial.getExpWeightedColumnData(columnName)
        columnData = self.padToPower2Length(columnData)
        signalData = columnData
        self.plot_spectrogram(signalData, 1/12, "Yearly Frequency Spectrum")



Test = True
if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    Test = False

if Test:
    finClass = FinancialDataClass('../data/financial_data.csv')
    finFftClass = FinancialFft(finClass)
    finFftClass.plotColumnSpectrum("Real_Price")  # , hanWindowSize=12*10)
    #finFftClass.plotAllColumns("Spectrum of all values")
    #finFftClass.plotColumn("Real_Price")#, hanWindowSize=12*10)
    #finFftClass.plotColumn("Real_Price", hanWindowSize=int(400/12)*12)
    #finFftClass.plotColumn("Real_Price", hanWindowSize=700)
