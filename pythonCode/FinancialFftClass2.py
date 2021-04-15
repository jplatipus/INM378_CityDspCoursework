from pythonCode.FinancialDataClass import FinancialDataClass
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
        self.columnData = self.financial.getExpWeightedColumnData("Real_Price")
        self.columnData = np.log10(self.columnData)
        #self.columnData = self.padToPower2Length(self.columnData)


    def displaySummaryInfo(self):
        self.financial.displayDataSummary()

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

    def lookForPeriodic(self, months):
        excerpt = self.columnData[:months]
        avg = np.mean(excerpt)
        norm_excerpt = excerpt - avg
        harmonic_model = fft.rfft(norm_excerpt)

        plt.figure()
        plt.plot(np.abs(harmonic_model))
        plt.title("Harmonic model")
        plt.show()

        threshold = 5
        predict_spectrum = np.where(np.abs(harmonic_model) < threshold, 0, harmonic_model)
        plt.figure()
        plt.plot(np.abs(predict_spectrum))
        plt.title('Prediction spectrum')
        plt.show()

        prediction = fft.irfft(predict_spectrum)

        test_length = 2  # expressed as number of times the excerpt length
        total_length = (test_length + 1) * months

        true_signal = self.columnData[:total_length] - avg
        periodic_prediction = np.tile(prediction, test_length + 1)

        plt.figure()
        plt.plot(true_signal, label='true signal')
        plt.plot(norm_excerpt, label='used excerpt')
        plt.plot(periodic_prediction, label='periodic prediction')
        plt.legend(loc='upper left')
        plt.show()
        unseenData = true_signal[months:]
        predictData = periodic_prediction[-len(unseenData):]
        rmse = np.sqrt(np.mean(np.power(unseenData - predictData, 2)))
        print(rmse)

        for threshold in np.arange(10, 20, 30):
            for years in np.arange(1, 8):
                excerpt_length = 12 * years
                excerpt = self.columnData[:excerpt_length]
                avg = np.mean(excerpt)
                norm_excerpt = excerpt - avg
                harmonic_model = fft.rfft(norm_excerpt)
                predict_spectrum = np.where(np.abs(harmonic_model) < threshold, 0, harmonic_model)
                prediction = fft.irfft(predict_spectrum)

                test_length = 3  # expressed as number of times the excerpt length
                total_length = (test_length + 1) * excerpt_length
                true_signal = self.columnData[:total_length] - avg
                periodic_prediction = np.tile(prediction, test_length + 1)
                rmse = np.sqrt(
                    np.mean(np.power(true_signal[excerpt_length:] - periodic_prediction[excerpt_length:], 2)))

                plt.figure()
                plt.plot(true_signal, label='true signal')
                plt.plot(norm_excerpt, label='used excerpt')
                plt.plot(periodic_prediction, label='periodic prediction')
                plt.title('The RMSE for threshold {} and a {}-Year excerpt is {:.2f}'.format(threshold, years, rmse))
                plt.legend(loc='lower right')
                plt.show()

finClass = FinancialDataClass('../data/financial_data.csv')
finFftClass = FinancialFf2t(finClass)
finFftClass.displaySummaryInfo()
finFftClass.lookForPeriodic(12*60)
