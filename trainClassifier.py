import os
import numpy as np
from scipy.signal import *
import csv
import matplotlib.pyplot as plt

from scipy import signal
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from sklearn.cluster import KMeans


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
    #Options to read: 'EEG-IO', 'EEG-VV', 'EEG-VR', 'EEG-MB'
    data_folder = 'EEG-IO' 

    # Parameters and bandpass filtering
    fs = 250.0

    # Reading data files
    file_idx = 0
    list_of_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) and '_data' in f] #List of all the files, Lists are randomized, its only looking for file with _data in it
    print(list_of_files)
    file_sig = list_of_files[file_idx] # Data File
    file_stim = list_of_files[file_idx].replace('_data','_labels') #Label File, Replacing _data with _labels
    print ("Reading: ", file_sig, file_stim)

    # Loading data
    if data_folder == 'EEG-IO' or data_folder == 'EEG-MB':
        data_sig = np.loadtxt(open(os.path.join(data_folder,file_sig), "rb"), delimiter=";", skiprows=1, usecols=(0,1,2)) #data_sig would be a buffer
    elif data_folder == 'EEG-VR' or data_folder == 'EEG-VV':
        data_sig = np.loadtxt(open(os.path.join(data_folder,file_sig), "rb"), delimiter=",", skiprows=5, usecols=(0,1,2)) 
        data_sig = data_sig[0:(int(200*fs)+1),:] # getting data ready -- not needed for previous 2 datasets
        data_sig = data_sig[:,0:3] #
        data_sig[:,0] = np.array(range(0,len(data_sig)))/fs


    ############ Calculating PSD ############
    index, ch = data_sig.shape[0], data_sig.shape[1]
    # print(index)
    feature_vectors = [[], []]
    feature_vectorsa = [[], []]


    x=1
    while x>0 and x<3:
        if x==1:
            data_sig[:,1] = lowpass(data_sig[:,1], 10, fs, 4)

        elif x==2:
            data_sig[:,2] = lowpass(data_sig[:,2], 10, fs, 4)

        for y in range(500, 19328 ,500):
            #print(ch)
            if x==1:
                DataFilter.detrend(data_sig[y-500:y, 1], DetrendOperations.LINEAR.value)

                psd = DataFilter.get_psd_welch(data_sig[y-500:y, 1], nfft, nfft//2, 250,
                                        WindowFunctions.BLACKMAN_HARRIS.value)

                band_power_delta = DataFilter.get_band_power(psd, 1.0, 4.0)

                # Theta 4-8
                band_power_theta = DataFilter.get_band_power(psd, 4.0, 8.0)

                #Alpha 8-12
                band_power_alpha = DataFilter.get_band_power(psd, 8.0, 12.0)

                #Beta 12-30
                band_power_beta = DataFilter.get_band_power(psd, 12.0, 30.0)
                # print(feature_vectors.shape)

                feature_vectors[x].insert(y, [band_power_delta, band_power_theta, band_power_alpha, band_power_beta])
                feature_vectorsa[x].insert(y, [band_power_delta, band_power_theta])

            elif x==2:
                DataFilter.detrend(data_sig[y-500:y, 2], DetrendOperations.LINEAR.value)

                psd = DataFilter.get_psd_welch(data_sig[y-500:y, 2], nfft, nfft//2, 250,
                                        WindowFunctions.BLACKMAN_HARRIS.value)

                band_power_delta = DataFilter.get_band_power(psd, 1.0, 4.0)

                # Theta 4-8
                band_power_theta = DataFilter.get_band_power(psd, 4.0, 8.0)

                #Alpha 8-12
                band_power_alpha = DataFilter.get_band_power(psd, 8.0, 12.0)

                #Beta 12-30
                band_power_beta = DataFilter.get_band_power(psd, 12.0, 30.0)
                # print(feature_vectors.shape)

                # feature_vectorsc[x].insert(y, [band_power_delta, band_power_theta, band_power_alpha, band_power_beta])
                # feature_vectorsd[x].insert(y, [band_power_delta, band_power_theta])

        x = x+1

    print(feature_vectorsa)
    powers = np.log10(np.asarray(feature_vectors, dtype=float))
    powers1 = np.log10(np.asarray(feature_vectorsa, dtype=float))
    # powers2 = np.log10(np.asarray(feature_vectorsb))
    # powers3 = np.log10(np.asarray(feature_vectorsc))
    print(powers.shape)
    print(powers1.shape)

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
    #Options to read: 'EEG-IO', 'EEG-VV', 'EEG-VR', 'EEG-MB'
    data_folder = 'EEG-IO' 

    # Parameters and bandpass filtering
    fs = 250.0

    # Reading data files
    file_idx = 0
    list_of_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) and '_data' in f] #List of all the files, Lists are randomized, its only looking for file with _data in it
    print(list_of_files)
    file_sig = list_of_files[file_idx] # Data File
    file_stim = list_of_files[file_idx].replace('_data','_labels') #Label File, Replacing _data with _labels
    print ("Reading: ", file_sig, file_stim)

    # Loading data
    if data_folder == 'EEG-IO' or data_folder == 'EEG-MB':
        data_sig = np.loadtxt(open(os.path.join(data_folder,file_sig), "rb"), delimiter=";", skiprows=1, usecols=(0,1,2)) #data_sig would be a buffer
    elif data_folder == 'EEG-VR' or data_folder == 'EEG-VV':
        data_sig = np.loadtxt(open(os.path.join(data_folder,file_sig), "rb"), delimiter=",", skiprows=5, usecols=(0,1,2)) 
        data_sig = data_sig[0:(int(200*fs)+1),:] # getting data ready -- not needed for previous 2 datasets
        data_sig = data_sig[:,0:3] #
        data_sig[:,0] = np.array(range(0,len(data_sig)))/fs


    ############ Calculating PSD ############
    index, ch = data_sig.shape[0], data_sig.shape[1]
    # print(index)
    feature_vectors = [[], []]
    feature_vectorsa = [[], []]


    x=1
    while x>0 and x<3:
        if x==1:
            data_sig[:,1] = lowpass(data_sig[:,1], 10, fs, 4)

        elif x==2:
            data_sig[:,2] = lowpass(data_sig[:,2], 10, fs, 4)

        for y in range(500, 19328 ,500):
            #print(ch)
            if x==1:
                DataFilter.detrend(data_sig[y-500:y, 1], DetrendOperations.LINEAR.value)

                psd = DataFilter.get_psd_welch(data_sig[y-500:y, 1], nfft, nfft//2, 250,
                                        WindowFunctions.BLACKMAN_HARRIS.value)

                band_power_delta = DataFilter.get_band_power(psd, 1.0, 4.0)

                # Theta 4-8
                band_power_theta = DataFilter.get_band_power(psd, 4.0, 8.0)

                #Alpha 8-12
                band_power_alpha = DataFilter.get_band_power(psd, 8.0, 12.0)

                #Beta 12-30
                band_power_beta = DataFilter.get_band_power(psd, 12.0, 30.0)
                # print(feature_vectors.shape)

                feature_vectors[x].insert(y, [band_power_delta, band_power_theta, band_power_alpha, band_power_beta])
                feature_vectorsa[x].insert(y, [band_power_delta, band_power_theta])

            elif x==2:
                DataFilter.detrend(data_sig[y-500:y, 2], DetrendOperations.LINEAR.value)

                psd = DataFilter.get_psd_welch(data_sig[y-500:y, 2], nfft, nfft//2, 250,
                                        WindowFunctions.BLACKMAN_HARRIS.value)

                band_power_delta = DataFilter.get_band_power(psd, 1.0, 4.0)

                # Theta 4-8
                band_power_theta = DataFilter.get_band_power(psd, 4.0, 8.0)

                #Alpha 8-12
                band_power_alpha = DataFilter.get_band_power(psd, 8.0, 12.0)

                #Beta 12-30
                band_power_beta = DataFilter.get_band_power(psd, 12.0, 30.0)
                # print(feature_vectors.shape)

                # feature_vectorsc[x].insert(y, [band_power_delta, band_power_theta, band_power_alpha, band_power_beta])
                # feature_vectorsd[x].insert(y, [band_power_delta, band_power_theta])

        x = x+1

    print(feature_vectorsa)
    powers = np.log10(np.asarray(feature_vectors, dtype=float))
    powers1 = np.log10(np.asarray(feature_vectorsa, dtype=float))
    # powers2 = np.log10(np.asarray(feature_vectorsb))
    # powers3 = np.log10(np.asarray(feature_vectorsc))
    print(powers.shape)
    print(powers1.shape)

    powers = powers.reshape(598, 4*2)
    powers = powers[:, 2:6]

    kmeans = KMeans(n_clusters=3, random_state=0)
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


def Kmeans2D():
def Kmeans4D():
    #Options to read: 'EEG-IO', 'EEG-VV', 'EEG-VR', 'EEG-MB'
    data_folder = 'EEG-IO' 

    # Parameters and bandpass filtering
    fs = 250.0

    # Reading data files
    file_idx = 0
    list_of_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) and '_data' in f] #List of all the files, Lists are randomized, its only looking for file with _data in it
    print(list_of_files)
    file_sig = list_of_files[file_idx] # Data File
    file_stim = list_of_files[file_idx].replace('_data','_labels') #Label File, Replacing _data with _labels
    print ("Reading: ", file_sig, file_stim)

    # Loading data
    if data_folder == 'EEG-IO' or data_folder == 'EEG-MB':
        data_sig = np.loadtxt(open(os.path.join(data_folder,file_sig), "rb"), delimiter=";", skiprows=1, usecols=(0,1,2)) #data_sig would be a buffer
    elif data_folder == 'EEG-VR' or data_folder == 'EEG-VV':
        data_sig = np.loadtxt(open(os.path.join(data_folder,file_sig), "rb"), delimiter=",", skiprows=5, usecols=(0,1,2)) 
        data_sig = data_sig[0:(int(200*fs)+1),:] # getting data ready -- not needed for previous 2 datasets
        data_sig = data_sig[:,0:3] #
        data_sig[:,0] = np.array(range(0,len(data_sig)))/fs


    ############ Calculating PSD ############
    index, ch = data_sig.shape[0], data_sig.shape[1]
    # print(index)
    feature_vectors = [[], []]
    feature_vectorsa = [[], []]


    x=1
    while x>0 and x<3:
        if x==1:
            data_sig[:,1] = lowpass(data_sig[:,1], 10, fs, 4)

        elif x==2:
            data_sig[:,2] = lowpass(data_sig[:,2], 10, fs, 4)

        for y in range(500, 19328 ,500):
            #print(ch)
            if x==1:
                DataFilter.detrend(data_sig[y-500:y, 1], DetrendOperations.LINEAR.value)

                psd = DataFilter.get_psd_welch(data_sig[y-500:y, 1], nfft, nfft//2, 250,
                                        WindowFunctions.BLACKMAN_HARRIS.value)

                band_power_delta = DataFilter.get_band_power(psd, 1.0, 4.0)

                # Theta 4-8
                band_power_theta = DataFilter.get_band_power(psd, 4.0, 8.0)

                #Alpha 8-12
                band_power_alpha = DataFilter.get_band_power(psd, 8.0, 12.0)

                #Beta 12-30
                band_power_beta = DataFilter.get_band_power(psd, 12.0, 30.0)
                # print(feature_vectors.shape)

                feature_vectors[x].insert(y, [band_power_delta, band_power_theta, band_power_alpha, band_power_beta])
                feature_vectorsa[x].insert(y, [band_power_delta, band_power_theta])

            elif x==2:
                DataFilter.detrend(data_sig[y-500:y, 2], DetrendOperations.LINEAR.value)

                psd = DataFilter.get_psd_welch(data_sig[y-500:y, 2], nfft, nfft//2, 250,
                                        WindowFunctions.BLACKMAN_HARRIS.value)

                band_power_delta = DataFilter.get_band_power(psd, 1.0, 4.0)

                # Theta 4-8
                band_power_theta = DataFilter.get_band_power(psd, 4.0, 8.0)

                #Alpha 8-12
                band_power_alpha = DataFilter.get_band_power(psd, 8.0, 12.0)

                #Beta 12-30
                band_power_beta = DataFilter.get_band_power(psd, 12.0, 30.0)
                # print(feature_vectors.shape)

                # feature_vectorsc[x].insert(y, [band_power_delta, band_power_theta, band_power_alpha, band_power_beta])
                # feature_vectorsd[x].insert(y, [band_power_delta, band_power_theta])

        x = x+1

    print(feature_vectorsa)
    powers = np.log10(np.asarray(feature_vectors, dtype=float))
    powers1 = np.log10(np.asarray(feature_vectorsa, dtype=float))
    # powers2 = np.log10(np.asarray(feature_vectorsb))
    # powers3 = np.log10(np.asarray(feature_vectorsc))
    print(powers.shape)
    print(powers1.shape)

    powers = powers.reshape(598, 4*2)
    powers = powers[:, 3:5]

    kmeans = KMeans(n_clusters=3, random_state=0)
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
