# Home_Automation_Using_Brain_Signals

Objective is controling home appliances with our brain. We are using brain signals in form of electroencephalogram (EEG)
to do this. We are analyzing data collected from various subjects to train various machine learning models namely SVM and Long-short term memory.

In our very first stage of project we have collected data only for one home appliance (Light Bulb).
We used Emotiv Epoc to collect EEG data of 4 subjects.

Subjects were asked to think to turn-on/turn-off/do-nothing lights while the EEG data is getting recorded.
We then applied noise filter and several band-pass filters in most common frequncy bands namely: alpha, beta, theta, delta and gamma.

We trained these 3-class data from 4 subjects,6 trials each, on 6-layer LSTM and found the best results on alpha-band:

                Accuracy    
Alpha-band:       50%
Theta-band:       43%

We are still collecting data. Once we collect enough data for Light-Blub, we will collect data for several other appliances.

## References
[1] Fabien Lotte. A Tutorial on EEG Signal Processing Techniques for Mental State Recognition in
Brain-Computer Interfaces. Eduardo Reck Miranda; Julien Castet. Guide to Brain-Computer
Music Interfacing, Springer, 2014. <hal-01055103>

[2] M. Rajya Lakshmi, Dr. T. V. Prasad, and Dr. V. Chandra Prakash, “Survey on EEG Signal Processing Methods”, IJARCSSE, 1(4):84-91, 2014.
