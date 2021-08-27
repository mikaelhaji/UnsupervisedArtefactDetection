


<img width="1000" alt="Screen Shot 2021-07-31 at 11 09 02 PM" src="https://user-images.githubusercontent.com/68840767/131073876-d1868c79-11b9-46ef-87d3-adba13d8e54c.png">



# **Abstract**

Electroencephalography (EEG) is used to diagnose, monitor, and prognosis various neurological diseases.  One of the most difficult aspects of EEG data is its susceptibility to a wide range of non-stationary noises caused by physiological, movement, and equipment artefacts. Existing artefact detection methods are inadequate because they need specialists to manually examine and annotate data for artefact parts. Existing artefact correction or removal methods are ineffective.  In this paper, proposed is a EEG noise-reduction technique that uses an unsupervised learning algorithm to identify eye blinks and jaw clench artefacts. The approach extracts important features and uses an unsupervised detection algorithm to identify EEG artefacts that are particular to a given task and individual. By making the approach, code, and data public, offered is a tool that is both immediately useful and may serve as an essential basis for future research in this area.


# **Demo Video**
https://user-images.githubusercontent.com/68840767/131074065-31848417-4ed9-4b8b-8aa7-d8784228a259.mp4



# **1.0 ~ Introduction**

Electroencephalography (EEG) equipment are widely utilised in clinical research, teaching, entertainment, and a wide range of other applications. However, most EEG applications are still restricted by the poor signal-to-noise ratio inherent in EEG data. Movement artefacts, physiological artefacts (e.g., sweat), and instrument artefacts are all causes of EEG noise (resulting from the EEG device itself). While researchers have developed a variety of techniques for identifying particular instances of these artefacts in EEG data, the majority of these methods involve manual tagging of example artefact segments or additional hardware, such as Electrooculography electrodes implanted around the eyes.

Manual annotation of artefacts in EEG data is difficult because it takes time and may be impossible if the particular profiles of artefacts in EEG data change as a function of the task, the person, or the experimental trial within a given task for a given subject, as they often do. These facts rapidly escalate the complexity of the artefact annotation issue, rendering a one-size-fits-all artefact identification approach impractical for many practical use cases.

One significant difficulty of artefact identification in EEG processing is that the concept of “artefact” varies depending on the job at hand. That is, an EEG segment is considered an artefact if and only if it degrades the performance of downstream techniques by appearing as uncorrelated noise in a feature space relevant to those methods. Muscle movement signatures, for example, confuse comma-prognostic categorization yet are helpful characteristics for identifying sleep stages.

The task-specific character of artefacts makes them particularly suited for detection utilising data-driven unsupervised techniques, since the only condition for identifying artefacts using such methods is that the artefacts be relatively rare. In other words, when we map our data into feature spaces relevant to the particular EEG task, artefacts should show out as uncommon abnormalities. Indeed, several cutting-edge techniques use unsupervised algorithms for detecting particular artefact types under specified conditions.

EEG artifact removal is one instance of a more general class of noise reduction problems. The removal of noise from signal data has been a topic of scientific inquiry since the 1940s; over the years, multiple signal processing approaches to this problem have found their way into EEG research. One such technique for artifact removal that is ubiquitous for EEG processing is Independent Component Analysis (ICA). This method and its modern derivative remain popular among the research community for unsupervised artifact correction. However, ICA still requires EEG experts to review the decomposed signals and manually classify them as either signal or noise.

Furthermore, while ICA is undeniably an invaluable tool for many EEG applications, it also has limitations that are particularly poignant when the number of channels is low; ICA can only extract as many independent components as there are channels and will therefore be unable to isolate all independent noise components if the total number of independent noise components and signal sources exceeds the number of EEG electrodes.

# **2.0 ~ Data Acquisition**

When conducting the recording sessions required for data collection for BCI devices, there are many difficulties to overcome. There were two headsets involved in this study. The primary headset tested was a modified version of the Ultracortex “Mark IV” headset from OpenBCI. This modification consisted of an added belt on the inside of the base of the headset. This helped to secure the headset to the user's head while reducing the impact of any movement on the electrode locations. The electrodes were the dry, spiky electrodes that came with the headset, and the headset was tightened so that it would not shift readily with minor movement by tightening the belt loop on the inner rim. The neck strap exerted downward stress on the spring-loaded electrodes, resulting in more contact and a reduced impedance of the dry electrodes. Finally, it was discovered that adding a little quantity of electrode gel further reduced the electrode impedance while avoiding many of the typical errors and problems associated with electrodes.

There were six datasets involved in this dataset, three recorded from the primary headset, the modified OpenBCI, and the three other recorded from a Muse. Below is a summary of what each dataset entailed:


<img width="629" alt="Screen Shot 2021-08-27 at 12 38 58 AM" src="https://user-images.githubusercontent.com/68840767/131072674-521771e5-3ee2-4111-83ff-85b4a6937691.png">

When dealing with this dataset, there are a few small details that bear mentioning as they influence design decisions.

1. Each recorded session was 10 minutes long.
2. For OpenBCI, there were 150,000 samples per session, meaning that the sample rate is 250 Hz
3. For Muse, there were 153600 samples per session, meaning that the sample rate is 256 Hz
4. There were 2 subjects involved in the data acquisition process. Subject 1 used OpenBCI to record his experiments and Subject 2 used Muse to record his experiments

In terms of acquiring data, a high-level uniform API interface known as Brainflow was used. The structure of the BrainFlow framework is split into 4 major modules, one of which being the data acquisition module. This module serves as an abstraction layer to gather data from biosensors. It presents reusable data structures and board classes from which a class for a certain board can be derived.

Following is a code example that shows how the final results for the Band Power approach are generated:

<img width="610" alt="Screen Shot 2021-08-27 at 12 42 11 AM" src="https://user-images.githubusercontent.com/68840767/131072847-2bed6f55-b44b-4e71-bc48-a8ed0a5a8b73.png">

# **3.0 ~ Unsupervised Artefact Detection**

### **3.1 Preprocessing**

***Detrending Time Series Data*** - A trend is a rise or decrease in the level of a time series over a lengthy period of time. Identifying and understanding trend information can aid in improving model performance. For starters, we can correct or remove the trend to simplify modeling and improve model performance. A time series with a trend is called non-stationary. An identified trend can be modeled. Once modeled, it can be removed from the time series dataset. This is called detrending the time series. If a dataset does not have a trend or we successfully remove the trend, the dataset is said to be trend stationary, which was the first step of preprocessing. [https://machinelearningmastery.com/time-series-trends-in-python/](https://machinelearningmastery.com/time-series-trends-in-python/)

***Bandpass Filter (5-50 Hz)*** - Signal processing design to have a frequency response as flat as possible as band-pass is known as Butterworth filter. A Band-pass filter passes any certain frequencies and it rejects all the frequencies outside of this band. So just cascading the high pass and low pass filter we can design this band-pass filter. In this study we applied a 2nd order 5 - 50 Hz Butterworth Bandwidth Filter in order to effectively remove a low-frequency noise from EEG, which is obtained by high power DC waves.

***Notch Filter (60 Hz)***

- In contrast to HFFs and LFFs, which have a progressive roll-off curve, the objective of a notch filter is to remove a particular frequency from an EEG signal. When the field of AC current from the electrical wiring and outlets that surround the patient contaminates the record, these notch filters come in handy. The transmission curve of an ideal notch filter has a flat response for all frequencies except the nominal frequency of the filter, where there is a notch in the curve indicating near perfect attenuation of any waves at that frequency. Of course, in the real world of functioning EEG, notch filters may not perform as well as this description suggests, but they usually do a decent job of reducing undesirable AC line voltage artefacts. As such, a 60 Hz Notch Filter was applied to the data.


<img width="1000" alt="Screen Shot 2021-08-27 at 12 43 18 AM" src="https://user-images.githubusercontent.com/68840767/131072939-7696d6dd-f256-4006-85d3-8b68862bb54a.png">

### **3.2 Feature Extraction**

***Calculate PSD Welch*** - One of the most widely used method to analyze EEG data is to decompose the signal into functionally distinct frequency bands, such as delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–12 Hz), beta (12–30 Hz), and gamma (30–100 Hz). This means that the EEG signal must be decomposed into frequency components, which is typically accomplished using Fourier transformations. The Fast Fourier Transform (FFT) is the almost universally used algorithm for computing the Fourier transform (and arguably the most important signal processing algorithm), which returns a complex number for each frequency bin from which the amplitude and phase of the signal at that frequency can be easily extracted. In spectral analysis, the magnitude-squared of the FFT is often used to determine the power spectral density, which is represented in microVolts^2 per Hertz in the case of EEG data.

Welch's periodogram, which consists of averaging successive Fourier transforms of tiny windows of the signal with or without overlapping, is the most commonly used technique for doing so, and was used in this project.

Welch's technique improves on the traditional periodogram's accuracy. The explanation is straightforward: EEG data are constantly time-varying, which means that if you look at 30 seconds of EEG data, the signal is extremely unlikely to appear like a perfect sum of pure sines. Rather, the spectral content of the EEG varies throughout time, influenced by neural activity underneath the skull. The problem is that, in order to provide a correct spectral estimate, a traditional periodogram needs the signal's spectral content to be stationary (i.e. time-unvarying) throughout the time period under consideration. The periodogram is usually skewed and includes much too much variation since this is never the case. Welch's technique significantly reduces this variation by averaging the periodograms acquired across small portions of the windows. However, this comes at the expense of a reduced frequency resolution. In fact, the frequency resolution is defined as follows:

<img width="1000" alt="Screen Shot 2021-08-27 at 12 44 51 AM" src="https://user-images.githubusercontent.com/68840767/131073069-6d597404-912a-4910-8390-006527441822.png">

where _**Fs**_ is the sampling frequency of the signal, _**N**_ the total number of samples and _**t**_ the duration, in seconds, of the signal.

### **3.3 The Model**

For the model, it was kept relatively simple by using an unsupervised K-Means Clustering algorithm that minimizes the distance of the points in a cluster with their centroid. The main objective of the K-Means algorithm is to minimize the sum of distances between the points and their respective cluster centroid. The first step of constructing the K Means algorithm is to choose the number of clusters, _**k**_. In this case, _**k**_ is equal to 3 **-**
- eye blink
- jaw clench
- rest

### **3.4 Accuracy Calculation**

The accuracy of a categorization is often used to assess its quality. It is also used in clustering. The scikit-learn accuracy score function, on the other hand, simply gives a bottom limit of accuracy for clustering.

Now, if we were actually classifying data then:

For ***n*** samples in a dataset, let ***yi*** be the class label for the ***i***-th sample and **ŷ***i* the predicted value. Accuracy between the class labels ***y*** and the predicted values ŷ is defined by:

<img width="1000" alt="Screen Shot 2021-08-27 at 12 50 17 AM" src="https://user-images.githubusercontent.com/68840767/131073522-b259c5da-e61d-4291-932e-dab2b6149adb.png">

where x→1(x) is the indicator function: 1(**ŷi**=**yi**)=1 if **ŷi**=**yi** and 0 else.

It is calculated as the total of the confusion matrix's diagonal elements divided by the number of samples to get a value between 0 and 1.

However, there is no connection given by the clustering method between the class labels and the anticipated cluster labels when it comes to clustering. As a result, the formula must be modified.

For clustering, we have to find the best match between the class labels and the cluster labels, so accuracy is defined by:

<img width="1000" alt="Screen Shot 2021-08-27 at 12 51 17 AM" src="https://user-images.githubusercontent.com/68840767/131073605-dac4a1a2-6726-47ac-acff-93aedabe5d16.png">

where ***P*** is the set of all permutations in [1;***K***] where ***K*** is the number of clusters.

It is important to note that there are ***K***! permutations in ***P*** which are quite high and the computation of accuracy is therefore prohibitive if we apply this formula blindly. The Hungarian algorithm was used to compute it in ***O(k^3)*** which is significantly faster.



