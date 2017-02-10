import csv
import os
from scipy.signal import savgol_filter
import numpy as np 
import re
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA


dir_name = './Cleaned_data'
file_names = [f for f in os.listdir(dir_name) if f.endswith('.csv')]
for file_name in file_names:
	all_data = [row for row in csv.reader(open(dir_name+'/'+file_name))]
	ids= re.findall('[\d]+',file_name)
	ids = [int(d) for d in ids]
	user_id = ids[0]
	trial_num = ids[1]
	# print all_data
	headers = all_data[0]
	all_data = all_data[1:]
	relevant_data = [row[3:17] for row in all_data]
	relevant_data = np.array(relevant_data)
	relevant_data = relevant_data.astype(float)

	# print file_name
	# print relevant_data[:10]
	# for i in range(len(relevant_data[0])):
	# 	data = relevant_data[:,i]
	# 	filtered_data = savgol_filter(data,5,2)
	# 	# print filtered_data.shape
	# 	# _ = raw_input("next")
	# 	# print np.subtract(data,filtered_data)
	# 	for j in range(len(all_data)):
	# 		all_data[j][i+3]=filtered_data[j]


	
	# relevant_data = [row[3:17] for row in all_data]
	# relevant_data = np.array(relevant_data)
	# relevant_data = np.transpose(relevant_data)
	print len(relevant_data),len(relevant_data[0])
	ica =  FastICA(n_components=3)
	new_features = ica.fit_transform(relevant_data)
	new_features = new_features.tolist()
	for i in range(len(new_features)):
		new_features[i] = [all_data[i][0]]+new_features[i]
	new_features = [headers[:1]+["new_feat"]*(len(new_features[0])-1)]+new_features
	print len(new_features),len(new_features[0])
	# _ = raw_input("next")
	with open("./Preprocessed_data_ica/subject_"+str(user_id)+"_trail_"+str(trial_num)+".csv",'wb') as resultFile:
	    wr = csv.writer(resultFile, dialect='excel')
	    wr.writerows(new_features)
	# data = relevant_data[:,0]
	# sp = np.fft.fft(data)
	# for i in range(len(sp)/2):
	# 	print sp[i+1],sp[-(i+1)],np.absolute(sp[i+1])

	# # plt.plot(sp)
	# # plt.show()
	# print sp
	# _ = raw_input("next")
	# freq = np.fft.fftfreq(data.shape[-1])
	# plt.plot(freq, sp.real, freq, sp.imag)	
	# plt.show()
