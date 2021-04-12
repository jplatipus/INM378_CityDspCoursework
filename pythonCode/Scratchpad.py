import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

SAMPLE_PICKLE_FILE = "sampleAcc.pkl"

def plotSampleSizeAccuracySearch():
    tempfile = open(SAMPLE_PICKLE_FILE, "rb")
    sampleSizeAccuracy = pkl.load(tempfile)
    tempfile.close()
    asArray = np.array(sampleSizeAccuracy)
    fig = plt.figure()
    plt.scatter(asArray[:,0], asArray[:,1])
    plt.show()

plotSampleSizeAccuracySearch()
arry = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])

print("Pad none")
print(np.pad(arry, [(0,0),(0,0)]))
print("insert row 1")
print(np.pad(arry, [(1,0), (0,0)], 'constant', constant_values=(0)))
print("append last row")
print(np.pad(arry, [(0,1), (0,0)], 'constant', constant_values=(0)))
print("insert 1st column")
print(np.pad(arry, [(0,0), (1,0)], 'constant', constant_values=(0)))
print("append last column")
print(np.pad(arry, [(0,0), (0,1)], 'constant', constant_values=(0)))
print("Pad all around")
print(np.pad(arry, [(1,1),(1,1)]))
