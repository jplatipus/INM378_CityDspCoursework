import math

import matplotlib.pyplot as plt
import numpy as np
import sys
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


    def replaceColumnNansWithValue(self, columnName, value):
        dataColumn = self.financial[columnName]
        copy = np.nan_to_num(dataColumn, copy=False, nan=value)
        self.financial[columnName] = copy

    def replaceColumnNansWithMax(self, columnName):
        dataColumn = self.financial[columnName]
        copy = np.nan_to_num(dataColumn, copy=True, nan=0)
        maxValue = np.max(copy)
        self.financial[columnName] = np.nan_to_num(dataColumn, copy=False, nan=maxValue)

    def displayDataSummary(self):
        print(self.financial.dtype.names)
        print("There are {} daily records in the file.".format(self.financial.size))

    def getColumnData(self, columnName):
        return self.financial[columnName]

    def getColumnName(self, columnIndex):
        return self.financial.dtype.names[columnIndex]

    def normaliseByColumn(self):
        columnNames = self.financial.dtype.names
        # don't normalise the date column:
        dataColumnIndeces = range(1, len(columnNames))
        for columnIndex in dataColumnIndeces:
            data = self.financial[columnNames[columnIndex]]
            normalized = self.normalize(data)
            self.financial[columnNames[columnIndex]] = normalized

    def normalize(self, data):
        min = np.min(data)
        max = np.max(data)
        euclidianLength = math.sqrt((min ** 2) + (max ** 2))
        data = data / euclidianLength
        return data

    # found on:
    # https://www.geeksforgeeks.org/how-to-create-a-single-legend-for-all-subplots-in-matplotlib/
    def displayLegendsAndShow(self, loc='upper left'):
        lines = []
        labels = []

        fig = plt.figure("MainFigure")
        addedLabels = set()
        for ax in fig.axes:
            axlines, axlabels = ax.get_legend_handles_labels()
            # print(Label)
            doBreak = False
            for axlabel in axlabels:
                if axlabel in addedLabels:
                    # label with the same name already added,
                    # we don't want to repeat
                    # subplot labels
                    doBreak = True
                    break
                addedLabels.add(axlabel)
            if doBreak:
                break
            lines.extend(axlines)
            labels.extend(axlabels)
            addedLabels.add(axlabel)
        fig.legend(lines, labels, loc=loc)
        plt.show()

    def plotPolyFitColumnOverTime(self, columnName, polyFitDegree, axis=None):
        date = self.financial['Date']
        columnData = self.financial[columnName]
        if axis == None:
            fig = plt.figure("MainFigure")
            ax = fig.subplots()
            plt.title('Evolution of {} over time'.format(columnName))
        else:
            ax = axis
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            coefficients = np.polyfit(date, columnData, polyFitDegree)
        #print("np.polyfit(x, y, 3): {}".format(coefficients))
        coeffPolynomial = np.poly1d(coefficients)
        datePoints = np.linspace(np.min(date), np.max(date), len(date))
        columnDataPoints = coeffPolynomial(datePoints)

        lines = []
        ax.plot(date, columnData, color="blue", label="Data")
        ax.plot(datePoints, columnDataPoints, '-', color='orange', alpha=0.5, label = "Polyfit")
        ax.plot(datePoints, columnData - columnDataPoints, '-', color='green', alpha=0.5, label="De-trended")
        if axis == None:
            self.displayLegendsAndShow()

    def plotLog10ColumnOverTime(self, columnName, axis=None):
        date = self.financial['Date']
        columnData = self.financial[columnName]
        columnDataLog10 = np.log10(columnData)
        if axis == None:
            fig = plt.figure("MainFigure")
            ax = fig.subplots()
            plt.title('Evolution of {} over time'.format(columnName))
        else:
            ax = axis
        ax.plot(date, columnData, color="blue", label="Data")
        logAxis = ax.twinx()
        logAxis.plot(date, columnDataLog10, '-', color='orange', alpha=0.5, label="Log10")
        ax.plot(date, columnData/columnDataLog10, '-', color='green', alpha=0.5, label="De-trended")
        if axis == None:
            self.displayLegendsAndShow()

    def plotExpWeightedColumnOverTime(self, columnName, axis=None):
        date = self.financial['Date']
        columnData = self.financial[columnName]
        columnFiltered = sig.lfilter([.25],[1,-0.75], columnData)
        if axis == None:
            fig = plt.figure("MainFigure")
            ax = fig.subplots()
            plt.title('Evolution of {} over time'.format(columnName))
        else:
            ax = axis
        ax.plot(date, columnData, color="blue", label="Data")
        logAxis = ax.twinx()
        ax.plot(date, columnFiltered, '-', color='orange', alpha=0.5, label = "Exp. Weighted")
        logAxis.plot(date, columnData - columnFiltered, '-', color='green', alpha=0.5, label="De-trended")
        if axis == None:
            self.displayLegendsAndShow()

    def plotAllColumns(self, title, ployfitDegree=None):
        dataColumnIndeces = range(0, len(self.financial.dtype.names))
        columns = 3
        rows = int(len(dataColumnIndeces) / 3 + int(len(dataColumnIndeces) % 3))
        fig = plt.figure("MainFigure")
        fig.subplots_adjust(hspace=0.6)
        fig.suptitle(title)
        date = self.financial['Date']
        for dataColumnIndex in range(0, len(dataColumnIndeces)):
            ax = fig.add_subplot(rows, columns, dataColumnIndex + 1)
            columnName = self.getColumnName(dataColumnIndex)
            if ployfitDegree == None:
                self.plotExpWeightedColumnOverTime(columnName, ax)
            else:
                self.plotPolyFitColumnOverTime(columnName, ployfitDegree, ax)
            ax.set(yticklabels=[])
            ax.set(xticklabels=[])
            if len(columnName) > 17:
                title = columnName[:17]
            else:
                title = columnName
            ax.title.set_text(title)
        self.displayLegendsAndShow(loc ='lower right')



Test = False
if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    Test = False

if Test:
    finClass = FinancialDataClass('../data/financial_data.csv')
    finClass.plotAllColumns("Linear Fit", 1)
    finClass.plotAllColumns("Log Fit")
    finClass.plotPolyFitColumnOverTime("Real_Price", 1)
    finClass.plotLog10ColumnOverTime("Real_Price")
    finClass.plotExpWeightedColumnOverTime("Real_Price")
