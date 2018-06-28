import numpy as np
import pandas as pd
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def loadTestData(filename):
    df = pd.read_csv(filename, header = None)
    return df


def generateWhiteNoise(length):
    wn = np.random.normal(0,1,length);
    return wn


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
   


if __name__ == '__main__':
    df = loadTestData('table_a.csv')
    plt.plot(df[5].values[800:])
    plt.show()
    close_prices = df[5].values[800:]
    imf = EEMD(close_prices, 300)
    for key in imf:
        #plot_imfs(close_prices, imf[key])
        plt.plot(hilbert(imf[key][:-2,:], axis = 0).T)
        plt.show()
