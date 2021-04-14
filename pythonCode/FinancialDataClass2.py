import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy.polynomial.polynomial as poly
import warnings
import scipy.signal as sig
from numpy.linalg import LinAlgError

warnings.simplefilter('ignore', np.RankWarning)

class FinancialDataClass:

    def __init__(self, dataFile):
        self.financial = np.genfromtxt(dataFile, delimiter=',', skip_header=2, names=True)

        #'Date',
        #'SP_Comp_P',
        #'Dividend_D',
        self.replaceColumnNansWithMax('Dividend_D')
        #'Earnings_E',
        self.replaceColumnNansWithMax('Earnings_E')
        #'Consumer_Price_Index_CPI',
        # 'Date_Fraction',
        # 'Long_Interest_Rate_GS10',
        # 'Real_Price',
        # 'Real_Dividend',
        self.replaceColumnNansWithMax('Real_Dividend')
        # 'Real_Earnings',
        self.replaceColumnNansWithMax('Real_Earnings')
        # 'Cyclically_Adjusted_Price_Earnings_Ratio_PE10_or_CAPE'
        self.replaceColumnNansWithValue('Cyclically_Adjusted_Price_Earnings_Ratio_PE10_or_CAPE', 13.0)
        #self.normaliseByColumn()
        self.rebaseData()
        self.financialOriginal = np.copy(self.financial)
        self.financialDetrend = self.detrendData()


    def replaceColumnNansWithValue(self, columnName, value):
        dataColumn = self.financial[columnName]
        copy = np.nan_to_num(dataColumn, copy=False, nan=value)
        self.financial[columnName] = copy

    def replaceColumnNansWithMax(self, columnName):
        dataColumn = self.financial[columnName]
        copy = np.nan_to_num(dataColumn, copy=True, nan=0)
        maxValue = np.max(copy)
        self.financial[columnName] = np.nan_to_num(dataColumn, copy=False, nan=maxValue)

    def rebaseData(self):
        columnNames = self.financial.dtype.names
        # don't normalise the date column:
        dataColumnIndeces = range(0, len(columnNames))
        for columnIndex in dataColumnIndeces:
            data = self.financial[columnNames[columnIndex]]
            min = np.min(data)
            if min == 0:
                continue
            if min < 0:
                data = data + abs(min)
            else:
                data = data - min
            self.financial[columnNames[columnIndex]] = data

    def detrendData(self):
        columnNames = self.financial.dtype.names
        financialCopy  = np.copy(self.financial)
        # don't normalise the date column:
        dataColumnIndeces = range(0, len(columnNames))
        for columnIndex in dataColumnIndeces:
            data = self.financial[columnNames[columnIndex]]
            #filteredData = sig.lfilter([.01], [1,-(1-.01)], data)
            filteredData = sig.lfilter([1/120,1/120,1/120,1/120], 1, data)
            financialCopy[columnNames[columnIndex]] = filteredData
        return financialCopy

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

    def getColumnData(self, columnName):
        return self.financial[columnName]

    def getColumnName(self, columnIndex):
        return self.financial.dtype.names[columnIndex]



    def plotAllDetrendedColumns(self, title, ployfitDegree=None):
        self.financial = self.financialDetrend
        self.plotAllColumns(title, ployfitDegree)
        self.financial = self.financialOriginal

    def plotAllColumns(self, title, ployfitDegree=None):
        dataColumnIndeces = range(0, len(self.financial.dtype.names))
        columns = 3
        rows = int(len(dataColumnIndeces) / 3 + int(len(dataColumnIndeces) % 3))
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.6)
        fig.suptitle(title)
        date = self.financial['Date']
        for dataColumnIndex in range(0, len(dataColumnIndeces)):
            ax = fig.add_subplot(rows, columns, dataColumnIndex + 1)
            columnName = self.getColumnName(dataColumnIndex)
            columnData = self.getColumnData(columnName)
            # plot the data
            ax.plot(date, columnData, 'b', label="Data")
            if columnName.endswith("CAPE"):
                ax.title.set_text("CAPE")
            else:
                ax.title.set_text(columnName)
            if ployfitDegree != None:
                self.tryPolyFit(ax, columnData, columnName, date, ployfitDegree)
            #ax.plot(date, regressY, 'g', alpha=0.5, label="Linear Trend")

            ax.set(yticklabels=[])
            ax.set(xticklabels=[])
        plt.show()

    def tryPolyFit(self, ax, columnData, columnName, date, polyFitDegree):
        try:
            print(columnName)
            #coefficients = poly.polyfit(range(0, len(date)), columnData, polyFitDegree, full=True)
            coefficients = poly.polyfit(date, columnData, polyFitDegree, full=True)
            print("Regress for degree {}: {}".format(polyFitDegree, coefficients))
            regressY = np.polyval(coefficients[0], range(0, len(date)))
            axTwin = ax.twinx()
            axTwin.plot(date, regressY*columnData, 'r', alpha=0.5, label="Linear Trend")
            return True
        except LinAlgError:
            print("LinAlgError for ", columnName)
            return False

    def tryPolyFitOld(self, ax, columnData, columnName, date, polyFitDegree):
        try:
            print(columnName)
            #coefficients = poly.polyfit(range(0, len(date)), columnData, polyFitDegree, full=True)
            coefficients = poly.polyfit(date, columnData, polyFitDegree, full=True)
            regressY = np.polyval(coefficients[0], range(0, len(date)))
            axTwin = ax.twinx()
            axTwin.plot(date, regressY*columnData, 'r', alpha=0.5, label="Linear Trend")
            return True
        except LinAlgError:
            print("LinAlgError for ", columnName)
            return False

Test = True
if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    Test = False

if Test:
    finClass = FinancialDataClass('../data/financial_data.csv')
    finClass.plotAllColumns("Original Data")
    finClass.plotAllDetrendedColumns("Detrended Data")
    '''
    finClass.plotAllDetrandedColumns("Quadratic Trend 2", 2)
    finClass.plotAllDetrandedColumns("Quadratic Trend 5", 5)
    finClass.plotAllDetrandedColumns("Quadratic Trend 10", 10)
    '''
    finClass.displayDataSummary()