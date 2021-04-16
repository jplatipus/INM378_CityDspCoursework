import math

import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
import scipy.signal as sig
from scipy import stats

from numpy.linalg import LinAlgError

warnings.simplefilter('ignore', np.RankWarning)

class FinancialDataClass:
    #
    # Class wraps the financial data file as a class, provides convenience methods to access the data, removes nan's,
    # provides plotting methods.


    # constructor: load the csv file, clean out nan's, replacing some with max values, others with a specified
    # value: Cyclically_Adjusted_Price_Earnings_Ratio_PE10_or_CAPE's nan's are set to 13, after inspecting the the data,
    # 13 is a good value for the adjusted PE ratio.
    # sets self.financial to the data read
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


    #
    # replaces nan values in the data column with the given value
    # columnName: the name of the column to process
    # value: the value that replaces the nan's
    def replaceColumnNansWithValue(self, columnName, value):
        dataColumn = self.financial[columnName]
        copy = np.nan_to_num(dataColumn, copy=False, nan=value)
        self.financial[columnName] = copy

    #
    # replaces nan values in the data column with the column's maximum value
    # columnName: the name of the column to process
    def replaceColumnNansWithMax(self, columnName):
        dataColumn = self.financial[columnName]
        copy = np.nan_to_num(dataColumn, copy=True, nan=0)
        maxValue = np.max(copy)
        self.financial[columnName] = np.nan_to_num(dataColumn, copy=False, nan=maxValue)

    # Display statsitical information for each column (the Date column is left out)
    def displayDataSummary(self):
        print(self.financial.dtype.names)
        print("There are {} daily records in the file.".format(self.financial.size))
        columnNames = self.financial.dtype.names
        # don't normalise the date column:
        dataColumnIndeces = range(1, len(columnNames))
        print("The file contains values that are read every month, {} months of data is {} years".format(self.financial.size, self.financial.size/12))
        for columnIndex in dataColumnIndeces:
            data = self.financial[columnNames[columnIndex]]
            normalized = self.normalize(data)
            self.financial[columnNames[columnIndex]] = normalized
            print(self.getColumnName(columnIndex))
            print("\tMean: {:.4f} Variance: {:.4f} Std Dev: {:.4f}\n\tMode: {:.4f} Median: {:.4f} Skewness: {:.4f}".format(
                np.mean(data), np.var(data), math.sqrt(np.var(data)), stats.mode(data)[0][0], np.median(data), stats.skew(data)))

    # get all the data for the requested column
    # columnNameOrIndex the column's name or index
    # return an array of the column data
    def getColumnData(self, columnNameOrIndex):
        if isinstance(columnNameOrIndex, str):
            return self.financial[columnNameOrIndex]
        columnNames = self.financial.dtype.names
        return self.getColumnData(columnNames[columnNameOrIndex])

    # get the name of the requested column's index
    # columnIndex the column's index
    # return the name of the column
    def getColumnName(self, columnIndex):
        return self.financial.dtype.names[columnIndex]

    # get a range array 0 .. the number of columns
    # columnIndex the column's index
    # return the name of the column
    def getColumnIndeces(self):
        return range(0, len(self.financial.dtype.names))

    # get the number of months the data spans
    # return the number of data rows (months)
    def getTotalMonthsInData(self):
        return len(self.getColumnData("Date"))

    # normalise the ALL columns' data (Date omitted), by column
    def normaliseByColumn(self):
        columnNames = self.financial.dtype.names
        # don't normalise the date column:
        dataColumnIndeces = range(1, len(columnNames))
        for columnIndex in dataColumnIndeces:
            data = self.financial[columnNames[columnIndex]]
            normalized = self.normalize(data)
            self.financial[columnNames[columnIndex]] = normalized

    # evaluate the normalisation of an array of data
    # data to normalise
    # return normalised data
    def normalize(self, data):
        min = np.min(data)
        max = np.max(data)
        euclidianLength = math.sqrt((min ** 2) + (max ** 2))
        data = data / euclidianLength
        return data

    # Similar to calling fig.lengend(), then fig.show():
    # Loops through the repeated labels created to extract the first set.
    # Sets the legend on the figure, so for subplots, only one legend is shown.
    # found on:
    # https://www.geeksforgeeks.org/how-to-create-a-single-legend-for-all-subplots-in-matplotlib/
    # then customised for this task, expects the current figure to be called "MainFigure"
    #
    # loc: location of the legend
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

    # calculate an exponentially weighted version of the column data
    # xDate: the column dates used in the calculation
    # yColumnData: the column data used in the calculation
    # polyFitDegree the weight to apply in the operation
    # return data fit
    def getPolyFitData(self, xDate, yColumnData, polyFitDegree):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
        coefficients = np.polyfit(xDate, yColumnData, polyFitDegree)
        coeffPolynomial = np.poly1d(coefficients)
        datePoints = np.linspace(np.min(xDate), np.max(xDate), len(xDate))
        columnDataPoints = coeffPolynomial(datePoints)
        return datePoints, columnDataPoints

    # perform a polyfit (linear or quadratic fit) of a column.
    # columnName the column to display
    # the degree for polyfit: 1 is linear, 2 or more is quadratic
    # axis the plot axis to display the plot,if none, a new figure is created
    def plotPolyFitColumnOverTime(self, columnName, polyFitDegree, axis=None):
        date = self.financial['Date']
        columnData = self.financial[columnName]
        if axis == None:
            fig = plt.figure("MainFigure")
            ax = fig.subplots()
            plt.title('Evolution of {} over time'.format(columnName))
        else:
            ax = axis

        datePoints, columnDataPoints = self.getPolyFitData(date, columnData, polyFitDegree)
        lines = []
        ax.plot(date, columnData, color="blue", label="Data")
        ax.plot(datePoints, columnDataPoints, '-', color='orange', alpha=0.5, label = "Polyfit")
        ax.plot(datePoints, columnData - columnDataPoints, '-', color='green', alpha=0.5, label="De-trended")
        if axis == None:
            self.displayLegendsAndShow()

    # calculate a log10 version of the column data
    # columnName the column used in the calculation
    # log10 of the data
    def getLog10ColumnData(self, columnName):
        columnData = self.financial[columnName]
        columnFiltered = np.loag10(columnData)
        return columnFiltered

    # Plot the log10 of a column's values.
    # columnName the column to display
    # axis the plot axis to display the plot,if none, a new figure is created
    def plotLog10ColumnOverTime(self, columnName, axis=None):
        date = self.financial['Date']
        columnDataLog10 = self.getLog10ColumnData(columnName)
        columnData = self.financial[columnName]
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

    # calculate an exponentially weighted version of the column data
    # columnName the column used in the calculation
    # the weight to apply in the operation
    # the data exponentially weighted
    def getExpWeightedColumnData(self, columnName, weight=0.25):
        columnData = self.financial[columnName]
        columnFiltered = sig.lfilter([weight],[1,weight - 1], columnData)
        return columnFiltered

    # Plot the log10 of a column's values.
    # columnName the column to display
    # axis the plot axis to display the plot,if none, a new figure is created
    def plotExpWeightedColumnOverTime(self, columnName, axis=None, weight=0.25):
        date = self.financial['Date']
        columnData = self.financial[columnName]
        columnFiltered = self.getExpWeightedColumnData(columnName, weight)
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

    # Plot all the columns data in a grid of plots. Either the exponentially weighted
    # data is overlayed or the linear / quadratic fie of the data is overlayed on the original data
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



Test = True
if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    Test = False

if Test:
    finClass = FinancialDataClass('../data/financial_data.csv')
    finClass.displayDataSummary()
    finClass.plotAllColumns("Linear Fit", 1)
    finClass.plotAllColumns("Quadratic Fit (2 degrees)", 2)
    finClass.plotAllColumns("Quadratic Fit (5 degrees)", 5)
    finClass.plotAllColumns("Exponential Fit")
    finClass.plotPolyFitColumnOverTime("Real_Price", 1)
    finClass.plotExpWeightedColumnOverTime("Real_Price")
