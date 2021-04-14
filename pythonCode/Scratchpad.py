import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy.polynomial.polynomial as poly
import warnings

financial = np.genfromtxt('../data/financial_data.csv', delimiter=',', skip_header=2, names=True)

date = financial['Date']
columnData = financial["Real_Price"]
fig = plt.figure()

x = date
y = np.nan_to_num(columnData, copy=False, nan=2297.11)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', np.RankWarning)
    z = np.polyfit(x, y, 30)
print("np.polyfit(x, y, 1): {}".format(z))

p = np.poly1d(z)
print("p(0.5): ", p(0.5))
with warnings.catch_warnings():
    warnings.simplefilter('ignore', np.RankWarning)
    p30 = np.poly1d(np.polyfit(x, y, 30))
print("p30(5): ", p30(5))

xp = np.linspace(1871.01, 2017.02, 100)
p_xp = p(xp)
p30_xp = p30(xp)

plt.plot(x, y, '.', xp, p30_xp, '--', xp, p_xp, '-')
plt.title('Evolution of Real_Price over time')
plt.show()