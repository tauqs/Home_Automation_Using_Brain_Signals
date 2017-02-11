import numpy as np
import random
from random import shuffle
from sklearn.decomposition import FastICA, PCA
import os
import csv
import other_training_models as models
import rnn_1
import rnn_2

header = ['label',  'AF3',  'F7',  'F3',  'FC5',  'T7',  'P7',  'O1',  'O2',  'P8',  'T8',  'FC6',  'F4',  'F8',  'AF4']  
columns = [0]+range(1,15)

#filenames = os.listdir('Processed_Data')
dir_name = 'Processed_Data'
#filenames=  os.listdir(dir_name)
## file naming format: subject_ID_trial_No; 8,16

#number_of_subjects = max(filenames,key=lambda x:x[8])

def calculate_subject_trial_count(filter_name):
    
    filenames = []
    ## dictionary to keep count of number of trials per subject_id
    subject_trial_count = {}
    if filter_name!="":
        filenames = [file for file in os.listdir(dir_name) if file.endswith(filter_name+'.csv')]
    else:
        filenames = [file for file in os.listdir(dir_name) if len(file.split('_'))==4]
    #print filenames
    
    ## loop to calculate subject_trial_count
    for file in filenames:
        if file[8] in subject_trial_count:
            subject_trial_count[file[8]]+=1
        else:
            subject_trial_count[file[8]]=1;
    return filenames,subject_trial_count

## function to load data
def load_data(filter_name = ""):
    train_data = []
    val_data = []
    test_data = []
    
    filenames,subject_trial_count = calculate_subject_trial_count(filter_name)
    
    for filename in filenames:
        file_source = os.path.join(dir_name,filename)
        f = open(file_source)
        f_csv = csv.reader(f)
        f_csv.next()                            
        temp_data = []       

        for row in f_csv:                                   
            req_row = [row[j] for j in columns]                       
            temp_data.append(req_row)

        ## condition to add the data as training data if it is the only 
        ## trial correspoding to that subject
        if subject_trial_count[(filename[8])]==1:
            train_data+=temp_data
        elif subject_trial_count[(filename[8])]==2:
            if int(filename[16])==subject_trial_count[(filename[8])]:
                test_data+=temp_data
            else:
                train_data+=temp_data
        else:
            if int(filename[16])==subject_trial_count[filename[8]]:
                test_data+=temp_data
            elif int(filename[16])==subject_trial_count[(filename[8])]-1:
                val_data+=temp_data
            else:
                train_data+=temp_data

        f.close()

    return train_data,val_data,test_data

## reformat data, such that it matches rnn input shape
def reformat_data(data,params,model='lstm'):
    data_length = len(data)
    print float(data_length)/(params['time_laps']*params['sampling_rate']) ,
    number_of_featured_data = int(data_length/(params['time_laps']*params['sampling_rate']))
    print number_of_featured_data
    input_length = int((params['num_cols']*params['time_laps']*params['sampling_rate']))
    if model =='lstm':
        input_length =  int(input_length/float(params['sequence_length'])) ## 14*384/6

    data_input = []
    data_output = []

    sec = params['time_laps']
    segment = int(sec*params['sampling_rate'])

    for i in range(number_of_featured_data):
        
        featured_data = data[i*segment:i*segment+segment]
        featured_data = np.array(featured_data)
        label = featured_data[0,0]
        featured_data = featured_data[::,1::]

        if model=='lstm':
            featured_data = featured_data.reshape((params['sequence_length'],input_length))
            output = [0,0,0]
            #print label
            if label == 'off_to_on':
                output[0] = 1
            elif label == 'on_to_off':
                output[1] = 1
            else:
                output[2] = 1                
            data_output.append(output)
        else:
            featured_data = featured_data.reshape((input_length))
            if label == 'off_to_on':
                #output[0] = 1
                data_output.append(0)
            elif label == 'on_to_off':
                #output[1] = 1
                data_output.append(1)
            else:
                #output[2] = 1
                data_output.append(2)

        data_input.append(featured_data)

    return data_input,data_output

def get_balanced_data(data):

    num_off_to_on = sum([d[0]=='off_to_on' for d in data])
    new_data = []
    count_do_nothing = 0

    for d in data:
        if d[0]=='do_nothing':
            if count_do_nothing<num_off_to_on:
                new_data.append(d)
                count_do_nothing+=1
        else:
            new_data.append(d)

    return new_data

def ICA(data,num_components):
    ica =  FastICA(n_components=num_components)
    new_features = ica.fit_transform(data)
    #new_features = new_features.tolist()
    return new_features

def get_fully_processed_data(other_params):

    train_data,val_data,test_data=None,None,None
    for filter_name,apply_ in filters.iteritems():
        if apply_:
            train_data_, val_data_, test_data_ = load_data(filter_name=filter_name)

            if balance_data:
                train_data_ = get_balanced_data(train_data_)
                val_data_ = get_balanced_data(val_data_)
                test_data_ =  get_balanced_data(test_data_)

            if train_data is None:
                train_data = np.array(train_data_)
                val_data = np.array(val_data_)
                test_data = np.array(test_data_)
            else:
                train_data = np.concatenate((train_data,np.array(train_data_)[::,1::]),axis=1)
                val_data = np.concatenate((val_data,np.array(val_data_)[::,1::]),axis=1)
                test_data = np.concatenate((test_data,np.array(test_data_)[::,1::]),axis=1)
        #train_data+=val_data

        #print np.array(train_data).shape
        #print train_data[0],train_data[1]
    #print train_data.shape
    if reduce_dimensionality and reduce_dimensionality_before_reshape:
        tmp = train_data[:,0]
        tmp = (np.reshape(tmp,(len(tmp),1)))
        print tmp
        train_data = ICA(train_data[::,1::],num_components)
        train_data = np.concatenate((tmp,train_data),axis=1)
        print train_data.shape
        
        tmp = val_data[:,0]
        tmp = (np.reshape(tmp,(len(tmp),1)))
        val_data = ICA(val_data[::,1::],num_components)
        val_data = np.concatenate((tmp,val_data),axis=1)
        
        tmp = test_data[:,0]
        tmp = (np.reshape(tmp,(len(tmp),1)))
        test_data = ICA(test_data[::,1::],num_components)
        test_data = np.concatenate((tmp,test_data),axis=1)

    train_data = train_data.tolist()
    val_data = val_data.tolist()
    test_data = test_data.tolist()

    train_input, train_output = reformat_data(train_data,params,model=model)
    val_input, val_output = reformat_data(val_data,params,model=model)
    test_input, test_output = reformat_data(test_data,params,model=model)

    
    if reduce_dimensionality and not(reduce_dimensionality_before_reshape):
        train_input = ICA(train_input,num_components*params['time_laps']*params['sampling_rate'])
        val_input = ICA(val_input,num_components*params['time_laps']*params['sampling_rate'])
        test_input = ICA(test_input,num_components*params['time_laps']*params['sampling_rate'])

    return train_input,train_output,val_input,val_output,test_input,test_output

def main():
    all_models = ['rnn_1','rnn_2','rfc','svm']
    model = all_models[0]
    train = bool(0)

    reduce_dimensionality = False
    balance_data = True
    reduce_dimensionality_before_reshape = True
    
    sequence_length = 4
    num_channels = 14
    time_laps = 0.125
    sampling_rate = 128
    model_specific_params ={}
    filters = {'alpha':True,'beta':True,'theta':True,'delta': False,'gamma':True}

    if model == 'rnn_1':
    	model_specific_params = {'epoch':1000,'batch_size':72, 'num_hidden':128}
    elif model == 'rnn_2':
    	model_specific_params = {'epoch':1000,'batch_size':72, 'num_hidden':128}
    elif model == 'rfc':
    	model_specific_params = {'n_estimators': 100}
    elif model == 'svm':
    	model_specific_params = {}

    num_filters = 0

    for key,val in filters.iteritems():
        if val:
            num_filters+=1

    num_components = 10*num_filters
    num_cols = num_channels*num_filters
    if reduce_dimensionality_before_reshape and reduce_dimensionality:
        num_cols = num_components

    params = {'num_channels':num_channels,'num_cols':num_cols, 
    'num_filters':num_filters,'time_laps':time_laps,'sampling_rate':sampling_rate,
    'sequence_length':sequence_length,'model_specific_params': model_specific_params}

    other_params = {'reduce_dimensionality':reduce_dimensionality,'balance_data':balance_data,
    'reduce_dimensionality_before_reshape':reduce_dimensionality_before_reshape, 'filters':filters}

    train_input,train_output,val_input,val_output,test_input,test_output = get_fully_processed_data(other_params)
    #print np.array(train_input).shape
    # print np.array(train_output).shape
    #print np.array(test_input).shape
    # print np.array(test_output).shape
    if model == 'rnn_1':
        rnn_1.main(params,train_input,train_output,val_input,val_output,test_input,test_output,train=train)
    elif model == 'svm':
        models.SVM(params,train_input,train_output,val_input,val_output,test_input,test_output)
    elif model == 'rfc':
        models.RFC(params,train_input,train_output,val_input,val_output,test_input,test_output)

if __name__ == '__main__':
    main()

