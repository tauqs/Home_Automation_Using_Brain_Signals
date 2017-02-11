# coding: utf-8
import numpy as np
import os
files = os.listdir('Time_Stamps')
files[0]
f = open(os.path.join('Time_Stamps',files[0]))
f = open(os.path.join('Time_Stamps',files[0]))
f
str = f.readline()
str
files.sort(key= lambda x: x[-3],x[-2])
files
files = sorted(files,key= lambda x: x[-3],x[-2])
files.sort(key= lambda x: x[-3],x[-2])
sorted(files,key= lambda x: x[-3],x[-2])
files = sorted(files,(key= lambda x: x[-3],x[-2]))
files = sorted(files,(key= lambda x: int(x[-3:-1])))
files = sorted(files,key= lambda x: int(x[-3:-1]))
files[0][-3:-1]
files = sorted(files,key= lambda x: int(x[-7:-5]))
files = sorted(files,key= lambda x: int(x[-8:-6]))
files = sorted(files,key= lambda x: int(x.split(' ')[-1].split('.')[0][1:-1]))
files
files = os.listdir('Time_Stamps')
files = sorted(files,key= lambda x: int(x.split(' ')[-1].split('.')[0][1:-1]))
files
f = open(os.path.join('Time_Stamps',files[0]))
f.readline()
f.readline()
f.readline()
f.seek(0)
f.seek(0)
f.readline()
f.seek(0)
time_stamps = []
for file in files:
    f = open(os.path.join('Time_Stamps',file))
    time_stamps.append(f.readline())
    f.close()
    
time_stamps
for file in files:
    f = open(os.path.join('Time_Stamps',file))
    str = f.readline()
    time_stamps.append(str)
    f.close()
        
time_stamps
time_stamps =[]
for file in files:
    f = open(os.path.join('Time_Stamps',file))
    str = f.readline()
    time_stamps.append(str)
    f.close()
        
time_stamps
time_stamps[0].split('\t')
time_stamps[0][:-2].split('\t')
time_stamps[0][:-1].split('\t')
time_stamps =[]
ids = []
trial_num = 1
for file in files:
    f = open(os.path.join('Time_Stamps',file))
    str = f.readline()
    vals = str[:-1].split('\t)
    values = {}
    values['id'] = int(vals[0])
    
    if int(vals[0]) in ids:
        trial_num+=1
    else:
        ids.append(int(vals[0]))
        trial_num=1
    values['trial_num'] = trial_num
    
    start_time = vals[-2]
    end_time = vals[-1]
    
    time_stamps.append(values)
    f.close()
    
        
time_stamps[0][:-1].split('\t')[-2].split(':')
time_stamps =[]
ids = []
trial_num = 1
for file in files:
    f = open(os.path.join('Time_Stamps',file))
    str = f.readline()
    vals = str[:-1].split('\t')
    values = {}
    values['id'] = int(vals[0])
    
    if int(vals[0]) in ids:
        trial_num+=1
    else:
        ids.append(int(vals[0]))
        trial_num=1
    values['trial_num'] = trial_num
    
    start_time = vals[-2].split(':')
    end_time = vals[-1].split(':')
    
    start_time[2] = start_time[2][:-3]
    end_time[2] = end_time[2][:-3]
    
    start_time = int(start_time[0])*60*60*1000 + int(start_time[1])*60*1000 + int(start_time[2])*1000 + int(start_time[3])
    end_time = int(end_time[0])*60*60*1000 + int(end_time[1])*60*1000 + int(end_time[2])*1000 + int(end_time[3])
    
    values['start_time'] = start_time
    values['end_time'] = end_time
    
    time_stamps.append(values)
    f.close()
    
time_stamps
csv_files =[if file.endswith('.csv') for file in os.listdir('.')]
csv_files =[if file.endswith('.csv'): file for file in os.listdir('.')]
csv_files =[file if file.endswith('.csv') for file in os.listdir('.')]
csv_files =[file if file.endswith('.csv') else pass for file in os.listdir('.')]
csv_files =[file if file.endswith('.csv') else '' for file in os.listdir('.')]
csv_files =[file if file.endswith('.csv') else []  for file in os.listdir('.')]
csv_files
import glob
csv_files = glob.glob('*.csv')
len(csv_files)
csv_files
import csv
f = open(csv_files[0])
f_csv = csv.reader(f)
f_csv
f_csv[0]
f_csv = list(f_csv)
f_csv[0]
f_csv[0][1].split(':')[-1].split(' ')
f_csv[1]
len(f_csv[1])
f_csv[1][22]
int(f_csv[1][22])
int(f_csv[1][22].split('.')[0])
int(f_csv[1][21].split('.')[0])
import time
time.ctime(os.path.getctime(os.path.join('Time_Stamps',files[0])))
files[0]
time.ctime(os.path.getmtime(os.path.join('Time_Stamps',files[0])))
time.mtime(os.path.getmtime(os.path.join('Time_Stamps',files[0])))
