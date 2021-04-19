from pythonCode.FinancialDataClass2 import FinancialDataClass2
import sys
import matplotlib.pyplot as plt
from scipy import fft, signal
import numpy as np
import math
from dsp_ap.operations import circ_convolve

class FinancialFft:
    # The class attempts to perform a Fourrier analysis of the data

    # constructor
    # financialClassInstance a FinancialDataClass instance with data preloaded
    def __init__(self, financialClassInstance):
        self.financial = financialClassInstance
        self.columnData = self.financial.deTrend("Real_Price", False)
        self.columnData, self.dates = self.padToPower2Length(self.columnData, self.financial.getColumnData("Date"))

    # make sure signal length is power of 2, pad with zero's if not to
    # make the length a power of 2
    def padToPower2Length(self, data, dates):
        length = len(data)
        while True:
            root = int(math.sqrt(length))
            if root ** 2 == length:
                break
            data = np.append(data, 0.0)
            dates = np.append(dates, dates[-1]+1)
            length = length + 1
        return data, dates


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
        plt.ylabel('frequency [months]')
        plt.xlabel('time [Years]')
        #     plt.xlim(0.5e7, 0.7e7) # adjust the x-axis to zoom in on a specific time region
        #     plt.xlim(5e7, 5.5e7)
        #     plt.ylim(0, 0.0005) # adjust the y-axis range to zoom in on low frequencies
        fig.show()

    # Plot a power spectrogram of the column data
    def plotColumnSpectrum(self, columnName):
        self.plot_spectrogram(self.columnData, 12, "Yearly Frequency Spectrum")

    # Look for periodicity (code taken from Lab 8, and modified)
    # startMonth the start month of the section to use for predicting
    # months: the number of months to extract, starting at startMonth, to use for trining
    # harmonic threshold: the harmonic model plot has several peak, be selecting a threshold, amplitudes below threshold
    #
    # Plots the frequencu spectrum of the selected data excerpt, plots the spectrum with the threshold filtered out,
    # uses the filtered spectrum to predict the data, which is plotted, in the third plot, along with the section extracted for
    # training, and the original data
    def lookForPeriodic(self, startMonth, months, harmonicThreshold):
        excerpt = self.columnData[startMonth:startMonth + months]
        dateExcerpts = self.dates[startMonth:startMonth + months]
        avg = np.mean(excerpt)
        normalisedExcerpt = excerpt - avg

        predictionSpectrum = self.plotHarmonicModels(harmonicThreshold, months, normalisedExcerpt)

        prediction = fft.irfft(predictionSpectrum)

        testLengthFactor = 2  # expressed as number of times the excerpt length
        totalLength = (testLengthFactor + 1) * months

        trueSignal = self.columnData[:totalLength] - avg
        periodic_prediction = np.tile(prediction, testLengthFactor + 1)

        plt.figure()

        plt.plot(trueSignal, label='true signal')
        excerptMonths = range(startMonth, startMonth + len(normalisedExcerpt))
        plt.plot(excerptMonths, normalisedExcerpt, label='used excerpt')
        # self.dates, periodic_prediction[0:len(self.dates)],
        plt.plot(periodic_prediction, label='periodic prediction')
        plt.legend(loc='upper left')
        plt.grid()

        plt.xlabel("Months since 1871")

        plt.show()
        unseenData = trueSignal[months:]
        predictData = periodic_prediction[-len(unseenData):]
        rmse = np.sqrt(np.mean(np.power(unseenData - predictData, 2)))
        print(rmse)

    #
    # Plots a harmonic model (frequencies in the  normalisedExcerpt)
    # and plots a spectrum with frequencies with an amplitude above the given threshold
    # return the spectrum of the frequencies' amplitudes above the given threshold
    def plotHarmonicModels(self, harmonicThreshold, months, normalisedExcerpt):
        harmonicModel = fft.rfft(normalisedExcerpt)
        plt.figure()
        plt.plot(np.abs(harmonicModel))
        plt.title("Harmonic model for months: 0..{}".format(months))
        plt.grid(axis="y")
        plt.xlabel("Month frequency")
        plt.show()
        threshold = harmonicThreshold
        predictionSpectrum = np.where(np.abs(harmonicModel) < threshold, 0, harmonicModel)
        plt.figure()
        plt.plot(np.abs(predictionSpectrum))
        plt.title("Prediction spectrum  for months: 0..{}".format(months))
        plt.grid()
        plt.show()
        return predictionSpectrum


finClass = FinancialDataClass2('../data/financial_data.csv')
finFftClass = FinancialFft(finClass)
#finFftClass.displaySummaryInfo()
finFftClass.lookForPeriodic(0, 608, 15)
finFftClass.lookForPeriodic(609, 533, 15)
finFftClass.lookForPeriodic(609 + 533, 415, 15)