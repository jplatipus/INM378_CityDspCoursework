from pythonCode.FinancialDataClass2 import FinancialDataClass2
import sys
import matplotlib.pyplot as plt
from scipy import fft, signal
import numpy as np
import math
from dsp_ap.operations import circ_convolve

#
#
class FinancialFf2t:

    def __init__(self, financialClassInstance):
        self.financial = financialClassInstance
        self.columnData = self.financial.deTrend("Real_Price", False)
        self.columnData, self.dates = self.padToPower2Length(self.columnData, self.financial.getColumnData("Date"))
        #self.columnData = np.log10(self.columnData)
        #self.columnData = self.padToPower2Length(self.columnData)


    def displaySummaryInfo(self):
        self.financial.displayDataSummary()

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

    def lookForPeriodic(self, startMonth, months):
        excerpt = self.columnData[startMonth:startMonth + months]
        dateExcerpts = self.dates[startMonth:startMonth + months]
        avg = np.mean(excerpt)
        normalisedExcerpt = excerpt - avg
        harmonicModel = fft.rfft(normalisedExcerpt)

        plt.figure()
        plt.plot(np.abs(harmonicModel))
        plt.title("Harmonic model for months: 0..{}".format(months))
        plt.grid(axis="y")
        plt.xlabel("Month frequency")
        plt.show()

        threshold = 10
        predictionSpectrum = np.where(np.abs(harmonicModel) < threshold, 0, harmonicModel)
        plt.figure()
        plt.plot(np.abs(predictionSpectrum))
        plt.title("Prediction spectrum  for months: 0..{}".format(months))
        plt.grid()
        plt.show()

        prediction = fft.irfft(predictionSpectrum)

        testLengthFactor = 2  # expressed as number of times the excerpt length
        totalLength = (testLengthFactor + 1) * months

        trueSignal = self.columnData[:totalLength] - avg
        periodic_prediction = np.tile(prediction, testLengthFactor + 1)

        plt.figure()
        plt.plot(self.dates, trueSignal, label='true signal')
        plt.plot(dateExcerpts, normalisedExcerpt, label='used excerpt')
        # self.dates, periodic_prediction[0:len(self.dates)],
        startIndex = startMonth
        length = min(len(self.dates), len(periodic_prediction))
        plt.plot(self.dates[startMonth:length], periodic_prediction[startMonth:length], label='periodic prediction')
        plt.legend(loc='lower left')
        plt.grid()
        plt.show()
        unseenData = trueSignal[months:]
        predictData = periodic_prediction[-len(unseenData):]
        rmse = np.sqrt(np.mean(np.power(unseenData - predictData, 2)))
        print(rmse)

finClass = FinancialDataClass2('../data/financial_data.csv')
finFftClass = FinancialFf2t(finClass)
#finFftClass.displaySummaryInfo()
finFftClass.lookForPeriodic(0, 12*60)
finFftClass.lookForPeriodic(30*12, 12*60)