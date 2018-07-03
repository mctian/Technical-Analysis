import numpy as np
import pandas as pd
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import time
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold


def loadTestData(filename):
    df = pd.read_csv(filename, header = None)
    return df


def generateWhiteNoise(length):
    wn = np.random.normal(0,.1,length);
    return wn


def longShortEncoding(currentTime, closePrices, highPrices, lowPrices, maxHoldTime, endPercent):
    timeEnd = currentTime + maxHoldTime
    highStopMagnitude = (endPercent + 1) * closePrices[currentTime]
    lowStopMagnitude = (1 - endPercent) * closePrices[currentTime]


    maxHoldStopReached = currentTime + maxHoldTime
    try:
        highStopReached = np.min(np.nonzero(highPrices[currentTime:timeEnd] >= np.ones(highPrices[currentTime:timeEnd].shape) * highStopMagnitude)) + currentTime
    except ValueError:
        highStopReached = maxHoldStopReached
    try:
        lowStopReached = np.min(np.nonzero(lowPrices[currentTime:timeEnd] <= np.ones(lowPrices[currentTime:timeEnd].shape) * lowStopMagnitude)) + currentTime
    except ValueError:
        lowStopReached = maxHoldStopReached
    firstStop = min(highStopReached, lowStopReached, maxHoldStopReached)
    if firstStop == maxHoldStopReached:
        return [0, firstStop]
    elif firstStop == highStopReached:
        return [1, firstStop]
    else:
        return [-1, firstStop]


def sampleWeightsByUniqueness(encodings):
    print(encodings)
    counts = np.zeros(encodings.shape[0])
    inverse_weight = np.zeros(encodings.shape[0])

    for i in range(encodings.shape[0]):
        counts[i:(encodings[i, 1] + 1)] += 1

    for i in range(encodings.shape[0]):
        inverse_weight[i] = np.mean(counts[i:(encodings[i, 1] + 1)])

    return 1/inverse_weight


def EEMD(sample, num_iterations):
    imf = {}
    for i in range(0,num_iterations):
        white_noise = generateWhiteNoise(len(sample))
        x = white_noise + sample
        decomp = EMD(x, maxiter = 10000)
        imfX = decomp.decompose()
        try:
            imf[imfX.shape[0]] += imfX
        except KeyError:
            imf[imfX.shape[0]] = imfX
    for key in imf:
        imf[key] /= key
    return imf


def rollingWindows(vector, length, startIndex, endIndex):
    indexes = np.arange(startIndex, endIndex)
    endIndexes = indexes + length
    xArr = np.matrix([vector[indexes[i]:endIndexes[i]] for i in range(0, len(indexes))])
    yArr = np.array([vector[startIndex + length:endIndex + length]]).T
    yArr = yArr.ravel()
    return xArr, yArr


def testWithIMFPrediction():
    t = time.time()
    df = loadTestData('table_bac.csv')
    plt.plot(df[5].values[:])
    #plt.show()
    close_prices = df[5].values[:]
    print(len(close_prices))
    close_prices = minmax_scale(close_prices)
    emd = EMD(close_prices, maxiter = 3000)
    imf = emd.decompose()
    plot_imfs(close_prices, imf)
    plt.plot(hilbert(imf, axis = 0).T)
    plt.show()
    svrlist = []
    predYVals = np.matrix([])
    for i in range(7,8):
        x,y = rollingWindows(imf[i], 500, 0, 2500)
        if i == 7:
            svr = svm.SVR(C = 0.1, cache_size = 4000)
        else:
            svr = svm.SVR(c = 10, cache_size = 4000)
        svr.fit(x,y)
        svrlist.append(svr)
        testX, testY = rollingWindows(imf[i], 500, 3040, 3400)
        predY = np.matrix(svr.predict(testX)).T
        print (predY.shape)
        try:
            predYVals = np.concatenate([predYVals, predY], axis = 1)
        except ValueError:
            predYVals = np.matrix(predY)
    svr = svm.SVR()
    svr.fit(imf[7:8,0:3000].T, close_prices[0:3000])
    predPrices = svr.predict(predYVals)
    print(mean_squared_error(close_prices[3540:3900], predPrices))
    print(mean_squared_error(close_prices[3540:3900], close_prices[3539:3899]))
    print(time.time() - t)


def optIMFPrediction():
    df = loadTestData('table_bac.csv')
    plt.plot(df[5].values[:])
    close_prices = df[5].values[:]
    close_prices = minmax_scale(close_prices)
    emd = EMD(close_prices, maxiter = 3000)
    imf = emd.decompose()
    svrlist = []
    predYVals = np.matrix([])
    tscv = TimeSeriesSplit(n_splits = 500)
    kf = KFold(n_splits = 10, shuffle = True)
    for i in range(imf.shape[0]):
        x,y = rollingWindows(imf[i], 500, 0, 3000)
        svr = svm.SVR(cache_size = 1000)
        parameters = {'C':[0.000001, 0.00001, 0.0001,0.001,0.01,0.1,1,10]}
        reg = GridSearchCV(svr, parameters, cv = kf, n_jobs = -1)
        reg.fit(x,y)
        print(reg.best_params_)
    return



if __name__ == '__main__':
    df = loadTestData('table_bac.csv')
    plt.plot(df[5].values[:])
    plt.plot(df[4].values[:])
    plt.plot(df[3].values[:])
    close_prices = df[5].values[:]
    low_prices = df[4].values[:]
    high_prices = df[3].values[:]
    encodings = np.array([longShortEncoding(i, close_prices, high_prices, low_prices, 100, 0.25) for i in range(3000)])

    weights = sampleWeightsByUniqueness(encodings) 
    print(weights)
