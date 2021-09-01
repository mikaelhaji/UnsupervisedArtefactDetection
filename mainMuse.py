import pyautogui
import numpy as np  # Module that simplifies computations on matrices
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
from scipy import signal
from sklearn.cluster import KMeans
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
import sys 
import os
from traindetect import Kmeans8D, Kmeans4D, Kmeans2D

# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL_BLINK = 1
INDEX_CHANNEL_JAW = 2
INDEX_CHANNELS = [INDEX_CHANNEL_BLINK, INDEX_CHANNEL_JAW]

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)

    # Get the stream info
    info = inlet.info()

    fs = int(info.nominal_srate())

    # ch_data = np.array([[], []])
    ch_data = [[], []]
    feature_vectors = [[], []]

    try:
        while True:

            for index in range(len(INDEX_CHANNELS)):

                """ 3.1 ACQUIRE DATA """
                eeg_data, timestamp = inlet.pull_chunk(
                    timeout=1, max_samples=int(252)) #possible 4 datapoint overlap

                # print(np.asarray(eeg_data)[:, int(INDEX_CHANNELS[index])].shape)
                # ch_data[index].append(DataFilter.detrend(np.asarray(eeg_data)[:, INDEX_CHANNELS[index]],  DetrendOperations.LINEAR.value)) 
                ch_data[index].append(np.asarray(eeg_data)[:, INDEX_CHANNELS[index]])
            
            if len(ch_data[0])>=3:

                ch_data_np = np.asarray(ch_data) #size of ch_data_np chages with ch_data
                ch, t, dp = ch_data_np.shape[0], ch_data_np.shape[1], ch_data_np.shape[2]
                ch_data_np = ch_data_np.reshape(ch, t*dp)
                ch, dp = ch_data_np.shape[0], ch_data_np.shape[1]

                for x in range(ch):
                    # DataFilter.detrend(ch_data_np[x, -256::], DetrendOperations.LINEAR.value) #full second overlap

                    f, Pxx_den = signal.welch(ch_data_np[x, -256::], fs=fs, nperseg=128, nfft=256) #figure out optimal window length 

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

                    feature_vectors[x] = [meanDelta, meanTheta, meanAlpha, meanBeta]

                powers = np.log10(np.asarray(feature_vectors))
                powers = powers.reshape(1, 2*4)
                powers = powers[:, 3:5]

                # print(powers)
                kmeans = Kmeans2D()
                label = kmeans.predict(powers)

                """ 3.3 COMPUTE NEUROFEEDBACK METRICS """
                print(label)

                if  label == 0:
                    print("""

                    right

                    """)
                    pyautogui.press('right')

                elif label == 1:
                    print("""

                    left

                    """)
                    pyautogui.press('left')

                elif label == 2:
                    print("""

                    up

                    """)
                    pyautogui.press('up')


    except KeyboardInterrupt:
        print('Closing!')
