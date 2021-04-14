import pythonCode.FinancialDataClass as FinancialDataClass
import sys
#
#
class FinancialFft:

    def __init__(self, financialClassInstance):
        self.financial = financialClassInstance



Test = True
if 'google.colab' in sys.modules or 'jupyter_client' in sys.modules:
    Test = False

if Test:
    finClass = FinancialDataClass('../data/financial_data.csv')
