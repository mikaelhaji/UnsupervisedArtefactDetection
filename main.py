import time
import brainflow
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations

BoardShim.enable_dev_board_logger () 
params = BrainFlowInputParams() 
board_id = BoardIds.CYTON_BOARD
sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)
params.serial_port = '/dev/cu.usbserial-DM03H7UU' # Port
board_id = 0 #BoardIds.CYTON_BOARD(0)
board = BoardShim(board_id, params) # Board Stream Object
board.prepare_session() # Prepare the session
# board.start_stream () # use this for default options
board.start_stream() #Create streaming thread
keep_alive = True

window_size = 1
sleep_time = 1

points_per_update = window_size * sampling_rate

# 0 = left ear, 1 = left forehead, 2 = Jaw Clenches , 3 = right ear
INDEX_CHANNEL_BLINK = [1]
INDEX_CHANNEL_JAW = [2]
INDEX_CHANNELS = [INDEX_CHANNEL_BLINK, INDEX_CHANNEL_JAW]


# index, ch = data.shape[0], df.shape[1]
feature_vectorsb = [[], []]
feature_vectorsc = [[], []]
data_np = []

nfft = 256

try:
    while keep_alive:
        time.sleep (sleep_time)
        data = board.get_current_board_data (int (points_per_update))
            # print ('Data Shape %s' % (str (data.shape)))

        data = data.tolist()

        data_np.append(data[1:3])
        

        if len(data_np)>2:
            newvar = data_np[-2::][:] 

            #ch, dp = data_np.shape[0], data_np.shape[1]
            newvar = np.asarray(newvar).reshape(2,2*250)

        if len(data_np)>=3:
            for x in range(2):
                DataFilter.detrend(newvar[x, -500::], DetrendOperations.LINEAR.value)


                DataFilter.perform_bandpass(newvar[x, -500::], 500, 250, 22, 3, FilterTypes.BESSEL.value,0)
                DataFilter.perform_bandstop(newvar[x, -500::], 500, 250, 60, 4, FilterTypes.BUTTERWORTH.value,0)
                # DataFilter.perform_bandpass(newvar[x, -500::], BoardShim.get_sampling_rate(board_id), 1.0, 1.0, 1,
                #                         FilterTypes.BESSEL.value, 0)
        
                # DataFilter.perform_bandstop(newvar[x, -500::], BoardShim.get_sampling_rate(board_id), 1.0, 1.0, 1,
                #                         FilterTypes.BUTTERWORTH.value, 0)

                # DataFilter.perform_bandpass(newvar[x, -500::], 250, 26, 50, 3, FilterTypes.BESSEL.value, 0)

                # DataFilter.perform_bandstop(newvar[x, -500::], 250, 60, 1, 4, FilterTypes.BUTTERWORTH.value, 0)


                    #30.0, 1.0, 3,
                    


                    #(1+51)/2), (51-1), 3, int filter_type, double ripple)


                psd = DataFilter.get_psd_welch(newvar[x, -500::], nfft, nfft//2, 250, 
                    WindowFunctions.BLACKMAN_HARRIS.value)

                #print (np.array(psd).shape)

                
                band_power_delta = DataFilter.get_band_power(psd, 1.0, 4.0)
                
                # Theta 4-8
                band_power_theta = DataFilter.get_band_power(psd, 4.0, 8.0)
                
                #Alpha 8-12
                band_power_alpha = DataFilter.get_band_power(psd, 8.0, 12.0)
                 
                #Beta 12-30
                band_power_beta = DataFilter.get_band_power(psd, 12.0, 30.0)
                    
                feature_vectorsb[x] = [band_power_delta, band_power_theta, band_power_alpha, band_power_beta]
                feature_vectorsc[x] = [band_power_delta, band_power_theta]

        powers = np.log10(np.asarray(feature_vectorsb))
        powers1 = np.log10(np.asarray(feature_vectorsc))
        
        #print(np.asarray(powers))
            # data = data[1:9,1::]
            # print (data.shape)#.shape) #2, 599, 4
        print (powers)



        # graph_data = 
        # lines = graph_data.split
        # print (feature_vectorsc)
            #print (powers)

    # board.stop_stream ()
    # board.release_session ()

except KeyboardInterrupt:
        print('Closing!')