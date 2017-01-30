import csv
import re
import os

def ms_since_noon(time_str):
	t = re.findall('[\d]+',time_str)
	t = [int(d) for d in t]
	if len(t)>3:
		t = t[3]+1000*(t[2]+60*(t[1]+60*(t[0]%12)))
	else:
		t = 1000*(t[2]+60*(t[1]+60*(t[0]%12)))
	return t
csv_file_name = 'Shubham Pandey 1.1.csv'
user_id = '1'
f = open(csv_file_name)

csv_file = csv.reader(f)
all_data = [row for row in csv_file]
header = all_data[0]
# print header
all_data = all_data[1:]


details = {}
for col in header:
	name = col.split(':')[0].strip()
	val = col.split(':')[-1].strip()
	details[name] = val
# print details

labels = details['labels'].split()

inception = details['recorded'].split()[-1]
inception = ms_since_noon(inception)

# 21 TIME_STAMP_s 4188
# 22 TIME_STAMP_ms 973
for i in range(len(all_data)):
	all_data[i][21]= int(float(all_data[i][21]))
	all_data[i][22]= int(float(all_data[i][22]))

ts_start_s = all_data[0][21]
ts_start_ms = all_data[1][21]
ts_start = ts_start_s*1000+ts_start_ms
print ts_start


somedir = './Time Stamps'
files = [f for f in os.listdir(somedir) if os.path.isfile(os.path.join(somedir, f))]
# print files



time_data = []
for fi in files:
	data = open(somedir+'/'+fi).read().strip()
	if data[0]==user_id:
		time_data.append([fi]+data.split('\t'))

from operator import itemgetter
time_data = sorted(time_data,key = itemgetter(-2))


trial_type = 0
labels = ['trial_type']+labels
final_data = [labels]
for data in time_data:
	trial_type+=1
	start_time = ms_since_noon(data[-2])-inception+ts_start
	end_time = ms_since_noon(data[-1])-inception+ts_start
	final_data = final_data+[[trial_type]+row for row in all_data if (1000*row[21]+row[22])>=start_time and (1000*row[21]+row[22])<=end_time]
	print data[0],data[-2],data[-1],start_time/1000,end_time/1000

# trial type : [1-12]
with open("output"+user_id+".csv",'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(final_data)