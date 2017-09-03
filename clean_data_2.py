import csv
import os
import re
import numpy as np

def ms_since_midnight(time_str):
	print time_str,
	t = re.findall('[\d]+',time_str)
	t = [int(d) for d in t]
	if 'PM' in time_str:
		t[0]+=12
	if (t[0]==12) and ('AM' in time_str) :
		t[0]=0
	if len(t)>3:
		t = t[3]+1000*(t[2]+60*(t[1]+60*t[0]))
	else:
		t = 1000*(t[2]+60*(t[1]+60*t[0]))
	print t
	return t

def diff_in_times(t1,t2):
	res = t1-t2
	print t1,t2,
	if res>=0:
		print res
		return res
	else:	
		print res+24*60*60*1000
		return res+24*60*60*1000

def main(IDs,source_folder='Collected_4th_March',target_dir='Data/Processed_Data_4_March'):

	raw_data_dir = 'Data/Smoothened_and_Digitally_Filtered_4_March'
	files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
	dir_name = 'Data/Raw_EEG_Data/'+source_folder+'/Time_Stamps'

	#target_dir = 'Data/Processed_Data'

	time_stamps_files = [f for f in os.listdir(dir_name)]
	time_data = [open(dir_name+'/'+file).read().strip().split('\t') for file in time_stamps_files]
	time_data = [[int(data[0]),data[-2],data[-1]]for data in time_data]
	user_time_dict = {}
	for data in time_data:
		if data[0] in user_time_dict.keys():
			user_time_dict[data[0]].append(data[1:])
		else:
			user_time_dict[data[0]] = [data[1:]]

	for user_id,time_stamps in user_time_dict.iteritems():
		user_time_dict[user_id] = [[ms_since_midnight(t[-2]),ms_since_midnight(t[-1])] for t in sorted(time_stamps)]
	# print sorted(time_data)
	print user_time_dict


	user_id_info = [row.split('ID') for row in open('user_id_list.txt').read().strip().split('\n')]
	user_id_info = [[row[0].strip(),int(row[-1])] for row in user_id_info if int(row[-1]) in IDs]
	print user_id_info

	user_files_dict = {}
	for info in user_id_info:
		user_files_dict[info[-1]] = sorted([file_name for file_name in files if info[0] in file_name])

	print user_files_dict

	for user_id,file_names in user_files_dict.iteritems():
		for trial in range(len(file_names)):
			
			csv_file_name = file_names[trial]
			
			tmp = csv_file_name.split('_')
			#if IDs == [2,5,6,7]:
			if (len(tmp)==4):
				trial_num = int(tmp[2])-1
			else:
				trial_num = int(tmp[2].split('.')[0])-1
			f_name = tmp[0]+'_'+tmp[1]+'_'+str(trial_num+1)+'.csv'
			#trial_num = int(tmp[2][0])-1
			#print csv_file_name,f_name,trial_num
			
			## we had one trial missing from user no. 3
			#if user_id==3:
			#	trial_num-=1
			
			filter_name = ""
			if(len(tmp)==4):
				filter_name = tmp[-1][0:-4] 

			print csv_file_name, f_name, trial_num, filter_name

			all_data = [row for row in csv.reader(open(os.path.join(raw_data_dir,csv_file_name)))]

			#csv_file_name_1 = csv_file_name[:-10]+'.csv'
			#header = next(csv.reader(open(os.path.join(raw_data_dir,csv_file_name_1))))
			all_data_1 = [row for row in csv.reader(open(os.path.join(raw_data_dir,f_name)))]
			
			header = all_data_1[0]
			details = {}
			for col in header:
				name = col.split(':')[0].strip()
				val = col.split(':')[-1].strip()
				details[name] = val
			labels = details['labels'].split()
			labels = ['label']+labels
			inception = details['recorded'].split()[-1]
			# print inception,
			inception = ms_since_midnight(inception)
			# print inception	

			all_data = all_data[1:]
			all_data_1 = all_data_1[1:]
			# 21 TIME_STAMP_s 4188
			# 22 TIME_STAMP_ms 973
			for i in range(len(all_data_1)):
				all_data_1[i][21]= int(float(all_data_1[i][21]))
				all_data_1[i][22]= int(float(all_data_1[i][22]))
			# print len(all_data),len(all_data[0])
			ts_start_s = all_data_1[0][21]
			ts_start_ms = all_data_1[0][22]
			ts_start = ts_start_s*1000+ts_start_ms
			inception+=ts_start_ms
			final_data = [labels]
			#trial_type = 0
			label_list = ['cut']*2+['off_to_on']*12+['cut']*3+ ['do_nothing']*8 + ['cut']*4
			label_list = label_list*2
			label_list += (['cut']*2+['on_to_off']*12+['cut']*3+ ['do_nothing']*8 + ['cut']*4)
			label_list += ['cut']*2+['on_to_off']*12
			label_iter = 0
			
			start_time = diff_in_times( user_time_dict[user_id][trial_num][0],inception )+ts_start
			#print '+',ts_start
			#print start_time
			next_label_time = start_time + 1000
			end_time = diff_in_times( user_time_dict[user_id][trial_num][1],inception )+ts_start
			print user_id,trial_num,start_time,end_time
			relevant_data = [row for (row,row_1) in zip(all_data,all_data_1) if (1000*row_1[21]+row_1[22])>=start_time and (1000*row_1[21]+row_1[22])<=end_time]
			#try:
			start_index = [x[0] for x in enumerate(all_data_1) if x[1][21]*1000+x[1][21]>start_time]
			#print start_index
			#print all_data_1[0][21],all_data_1[0][22]
			start_index = next(x[0] for x in enumerate(all_data_1) if x[1][21]*1000+x[1][22]>start_time)
			print start_index
			for label_iter in range(len(label_list)):
				if label_list[label_iter] not in ['cut']:
					final_data += [ [label_list[label_iter]]+row 
					for row in all_data[ (start_index+label_iter*128):(start_index+(label_iter+1)*128) ]]

			#except Exception, err:
			#	print Exception,err
			# os.chdir('./Cleaned_data')
			# final_data = [labels]+[final_data]
			output_file_name = "subject_"+str(user_id)+"_trial_"+str(trial_num+1)
			if filter_name!="":
				output_file_name+="_"+filter_name
			output_file_name+=".csv"

			if(len(tmp)==3):
				final_data = np.array(final_data)[::,[0]+range(3,17)]

			with open(target_dir+'/'+output_file_name,'wb') as resultFile:
			    wr = csv.writer(resultFile, dialect='excel')
			    wr.writerows(final_data)

if __name__ == '__main__':
	IDs = [3]+range(8,20)
	main(IDs,target_dir='Data/Processed_Data_4_March')
