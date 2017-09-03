import csv
import os
from scipy.signal import savgol_filter
import numpy as np 
import re

def main(folder='Collected_4th_March'):
	dir_name = 'Data/Raw_EEG_Data/'+folder
	file_names = [f for f in os.listdir(dir_name) if f.endswith('.csv')]
	target_dir = 'Data/Smoothened_Data_4_March/'

	for file_name in file_names:
		all_data = [row for row in csv.reader(open(dir_name+'/'+file_name))]
		#ids= re.findall('[\d]+',file_name)
		#ids = [int(d) for d in ids]
		#user_id = ids[0]
		#trial_num = ids[1]
		# print all_data
		headers = all_data[0]
		all_data = all_data[1:]
		relevant_data = [row[2:16] for row in all_data]
		relevant_data = np.array(relevant_data)
		relevant_data = relevant_data.astype(float)

		# print file_name
		# print relevant_data[:10]
		for i in range(len(relevant_data[0])):
			data = relevant_data[:,i]
			filtered_data = savgol_filter(data,5,2)
			print np.subtract(data,filtered_data)
			for j in range(len(all_data)):
				all_data[j][i+2]=filtered_data[j]
		all_data = [headers]+all_data
		#with open("./Smoothened_Data/subject_"+str(user_id)+"_trail_"+str(trial_num)+".csv",'wb') as resultFile:
		with open(target_dir+file_name,'wb') as resultFile:
			    wr = csv.writer(resultFile, dialect='excel')
			    wr.writerows(all_data)

if __name__ == '__main__':
	main()