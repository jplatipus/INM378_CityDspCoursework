import matplotlib.pyplot as plt
import numpy as np
import sys

class FinancialDataClass:

    def __init__(self, dataFile):
        self.financial = np.genfromtxt(dataFile, delimiter=',', skip_header=2, names=True)

    def displayDataSummary(self):
        print(self.financial.dtype.names)
        print("There are {} daily records in the file.".format(self.financial.size))

    def plotColumnOverTime(self, columnName):
        date = self.financial['Date']
        real_price = self.financial[columnName]
        plt.figure()
        plt.plot(date, real_price)
        plt.title('Evolution of {} over time'.format(columnName))
        plt.show()

    def plotAllColumns(self):
        for name in self.financial.dtype.names:
            self.plotColumnOverTime(name)



Test = True
if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    Test = False

if Test:
    finClass = FinancialDataClass('../data/financial_data.csv')
    #finClass.plotAllColumns()
    finClass.displayDataSummary()