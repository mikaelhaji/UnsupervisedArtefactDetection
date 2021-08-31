import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans

from brainflow.data_filter import DataFilter, DetrendOperations

def nextpow2(i):
    """
    Find the next power of 2 for number i
    """
    n = 1
    while n < i:
        n *= 2
    return n


def d_2d_to_3d(x, agg_num, hop):

    # Pad to at least one block. 
    len_x, n_in = x.shape
    if (len_x < agg_num): #not in get_matrix_data
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))

    # main 2d to 3d. 
    len_x = len(x) #15344
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1 : i1 + agg_num])
        i1 += hop

    return np.array(x3d)

def Kmeans8D():

    df = pd.read_csv(r'C:\Users\anush\OneDrive\Documents\HTN\BCIstuff\MixedBlink.csv', usecols=[2,3])

    # nfft = nextpow2(256)
    df = df.to_numpy()
    # print(df.shape)

    # plt.scatter(df[:, 0],df[:, 1], alpha=0.3,
    #             cmap='viridis')
    # plt.show()

    index, ch = df.shape[0], df.shape[1]
    feature_vectors = [[], []]

    # print(index, ch)

    for x in range(ch):
        for y in range(512,index,256):
            DataFilter.detrend(df[y-256:y, x], DetrendOperations.LINEAR.value)

            f, Pxx_den = signal.welch(df[y-260:y-4, x], fs=256, nperseg=128, nfft=256) #simulated 4 point overlap

            ind_delta, = np.where(f < 4)
            meanDelta = np.mean(Pxx_den[ind_delta], axis=0)
            # Theta 4-8
            ind_theta, = np.where((f >= 4) & (f <= 8))
            meanTheta = np.mean(Pxx_den[ind_theta], axis=0)
            # Alpha 8-12
            ind_alpha, = np.where((f >= 8) & (f <= 12))
            meanAlpha = np.mean(Pxx_den[ind_alpha], axis=0)
            # Beta 12-30
            ind_beta, = np.where((f >= 12) & (f < 30))
            meanBeta = np.mean(Pxx_den[ind_beta], axis=0)

            feature_vectors[x].insert(y, [meanDelta, meanTheta, meanAlpha, meanBeta])

    powers = np.log10(np.asarray(feature_vectors))
    # print(powers)
    # print(powers.shape)

    # plt.show()

    powers = powers.reshape(598, 4*2)

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(powers)
    return kmeans

    # clusters = kmeans.fit(powers)
    # centers = kmeans.cluster_centers_
    # covMatrix = np.cov(centers.T)
    # corMatrix = np.corrcoef(centers.T)
    # # sn.heatmap(covMatrix, annot=True, fmt='g')
    # # plt.show()
    # sn.heatmap(corMatrix, annot=True, fmt='g')
    # plt.show()

    # plt.scatter(powers[:, 2], powers[:, 3], c=clusters, s=50,
    #             cmap='viridis')

    # # print(kmeans.cluster_centers_[:, :])
    # plt.scatter(centers[:, 2], centers[:, 3], c='black', s=200, alpha=0.5)
    # plt.show()

def Kmeans4D():

    df = pd.read_csv(r'C:\Users\anush\OneDrive\Documents\HTN\BCIstuff\MixedBlink.csv', usecols=[2,3])

    # nfft = nextpow2(256)
    df = df.to_numpy()
    # print(df.shape)

    # plt.scatter(df[:, 0],df[:, 1], alpha=0.3,
    #             cmap='viridis')
    # plt.show()

    index, ch = df.shape[0], df.shape[1]
    feature_vectors = [[], []]

    # print(index, ch)

    for x in range(ch):
        for y in range(512,index,256):

            DataFilter.detrend(df[y-256:y, x], DetrendOperations.LINEAR.value)

            f, Pxx_den = signal.welch(df[y-260:y-4, x], fs=256, nperseg=128, nfft=256) #simulated 4 point overlap

            ind_delta, = np.where(f < 4)
            meanDelta = np.mean(Pxx_den[ind_delta], axis=0)
            # Theta 4-8
            ind_theta, = np.where((f >= 4) & (f <= 8))
            meanTheta = np.mean(Pxx_den[ind_theta], axis=0)
            # Alpha 8-12
            ind_alpha, = np.where((f >= 8) & (f <= 12))
            meanAlpha = np.mean(Pxx_den[ind_alpha], axis=0)
            # Beta 12-30
            ind_beta, = np.where((f >= 12) & (f < 30))
            meanBeta = np.mean(Pxx_den[ind_beta], axis=0)

            feature_vectors[x].insert(y, [meanDelta, meanTheta, meanAlpha, meanBeta])

    powers = np.log10(np.asarray(feature_vectors))
    # print(powers)

    powers = powers.reshape(598, 4*2)
    powers = powers[:, 2:6]
    # print(powers.shape)

    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(powers)
    return kmeans

    # clusters = kmeans.fit_predict(powers)
    # centers = kmeans.cluster_centers_
    # covMatrix = np.cov(centers.T)
    # corMatrix = np.corrcoef(centers.T)
    # # sn.heatmap(covMatrix, annot=True, fmt='g')
    # # plt.show()
    # sn.heatmap(corMatrix, annot=True, fmt='g')
    # plt.show()

    # plt.scatter(powers[:, 2], powers[:, 4], c=clusters, s=50,
    #             cmap='viridis')

    # # # print(kmeans.cluster_centers_[:, :])
    # plt.scatter(centers[:, 2], centers[:, 4], c='black', s=200, alpha=0.5)
    # plt.show()

def Kmeans2D():

    df = pd.read_csv(r'C:\Users\anush\OneDrive\Documents\HTN\BCIstuff\MixedBlink.csv', usecols=[2,3])

    # nfft = nextpow2(256)
    df = df.to_numpy()
    # print(df.shape)

    # plt.scatter(df[:, 0],df[:, 1], alpha=0.3,
    #             cmap='viridis')
    # plt.show()

    index, ch = df.shape[0], df.shape[1]
    feature_vectors = [[], []]

    # print(index, ch)

    for x in range(ch):
        for y in range(512,index,256):

            DataFilter.detrend(df[y-256:y, x], DetrendOperations.LINEAR.value)

            f, Pxx_den = signal.welch(df[y-260:y-4, x], fs=256, nperseg=128, nfft=256) #simulated 4 point overlap

            ind_delta, = np.where(f < 4)
            meanDelta = np.mean(Pxx_den[ind_delta], axis=0)
            # Theta 4-8
            ind_theta, = np.where((f >= 4) & (f <= 8))
            meanTheta = np.mean(Pxx_den[ind_theta], axis=0)
            # Alpha 8-12
            ind_alpha, = np.where((f >= 8) & (f <= 12))
            meanAlpha = np.mean(Pxx_den[ind_alpha], axis=0)
            # Beta 12-30
            ind_beta, = np.where((f >= 12) & (f < 30))
            meanBeta = np.mean(Pxx_den[ind_beta], axis=0)

            feature_vectors[x].insert(y, [meanDelta, meanTheta, meanAlpha, meanBeta])

    powers = np.log10(np.asarray(feature_vectors))
    # print(powers)

    powers = powers.reshape(598, 4*2)
    powers = powers[:, 3:5]
    # print(powers.shape)

    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(powers)
    return kmeans

    # clusters = kmeans.fit_predict(powers)
    # centers = kmeans.cluster_centers_
    # covMatrix = np.cov(centers.T)
    # corMatrix = np.corrcoef(centers.T)
    # # sn.heatmap(covMatrix, annot=True, fmt='g')
    # # plt.show()
    # sn.heatmap(corMatrix, annot=True, fmt='g')
    # plt.show()

    # plt.scatter(powers[:, 2], powers[:, 4], c=clusters, s=50,
    #             cmap='viridis')

    # # # print(kmeans.cluster_centers_[:, :])
    # plt.scatter(centers[:, 2], centers[:, 4], c='black', s=200, alpha=0.5)
    # plt.show()
