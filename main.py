import numpy as np
import random
from random import shuffle
from sklearn.decomposition import FastICA, PCA
from sklearn import preprocessing
from scipy.stats import mode
import os
import csv
import other_training_models as models
import rnn_1
import rnn_2
import rnn_rfc

header = ['label',  'AF3',  'F7',  'F3',  'FC5',  'T7',  'P7',  'O1',  'O2',  'P8',  'T8',  'FC6',  'F4',  'F8',  'AF4']  
columns = [0]+range(1,15)

#filenames = os.listdir('Processed_Data')
#dir_name = 'Data/Processed_Data'
dir_name = 'Data/Processed_Data'
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
        trial_ID = file.split('_')[1]
        if trial_ID in subject_trial_count:
            subject_trial_count[trial_ID]+=1
        else:
            subject_trial_count[trial_ID]=1;
    return filenames,subject_trial_count

## function to load data
def load_data(active_channels,ignore_symmetrical_correlation,filter_name = ""):
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
            if ignore_symmetrical_correlation:                                   
                req_row = [row[j] for j in columns if j==0 or active_channels[j-1]==1]
            else: 
                req_row = [row[j] for j in columns]                                              
            temp_data.append(req_row)

        ## condition to add the data as training data if it is the only 
        ## trial correspoding to that subject
        trial_ID = filename.split('_')[1]
        #if trial_ID=='2':
        trial_num = int(filename.split('_')[3].split('.')[0])
        if subject_trial_count[(trial_ID)]==1:
            train_data+=temp_data
        elif subject_trial_count[(trial_ID)]==2:
            if trial_num==subject_trial_count[(trial_ID)]:
                test_data+=temp_data
            else:
                train_data+=temp_data
        else:
            if trial_num==subject_trial_count[trial_ID]:
                test_data+=temp_data
            elif trial_num==subject_trial_count[trial_ID]-1:
                val_data+=temp_data
            else:
                train_data+=temp_data

        f.close()

    return train_data,val_data,test_data

def resample(data,sampling_rate,types):
    data_ = np.array(data)
    new_data = []
    dx = (128/sampling_rate)
    if dx == 1: return data 

    for i in range(len(data)/dx):
        rows  = data_[i*dx:i*dx+dx]
        row_ = []
        if types['avg']:
            row_ = np.mean(np.asarray(rows[::,1::],np.float),axis=0)
            #row_ = np.append(row_,row_)
        if types['max']:
            row = np.max(np.asarray(rows[::,1::],np.float64),axis=0)
            #print np.array(row).shape
            row_ = np.append(row_,row)
        if types['mode']:
            row = np.array(mode(np.asarray(rows[::,1::],np.float).astype(np.int64),axis=0))[0][0]
            #print np.array(row).shape
            row_ = np.append(row_,row)
        if types['rms']:
            row = np.sqrt(np.mean(np.square(np.asarray(rows[::,1::],np.float)),axis=0))
            row_ = np.append(row_,row)
        row = np.append(rows[0,0],row_)
        new_data.append(row.tolist())
    return new_data



## reformat data, such that it matches rnn input shape
def reformat_data(data,params,model='rnn_1'):
    print '\n\nnum_rows: ',len(data)
    data = resample(data,params['sampling_rate'],params['sampling_types'])
    data_length = len(data)
    print 'data shape: ',np.array(data).shape
    print 'num_rows after resampling:',data_length
    print 'num_featured_rows:', float(data_length)/(params['time_laps']*params['sampling_rate']) ,
    number_of_featured_data = int(data_length/(params['time_laps']*params['sampling_rate']))
    #print number_of_featured_data
    input_length = int((params['num_sampling_types']*params['num_cols']*params['time_laps']*params['sampling_rate']))
    num_filters = params['num_filters']
    num_channels = params['num_channels']
    if model.startswith('rnn'):
        input_length =  int(input_length/float(params['sequence_length'])) ## 14*384/6

    sec = params['time_laps']
    segment = int(sec*params['sampling_rate'])

    data_output = []
    data_input = []    

    for i in range(number_of_featured_data):
        
        featured_data = data[i*segment:i*segment+segment]
        featured_data = np.array(featured_data)
        label = featured_data[0,0]
        featured_data = featured_data[::,1::]

        if params['ignore_3rd_class'] and label == 'do_nothing':
            continue

        if model.startswith('rnn'):
            if model == 'rnn_1':
                featured_data = featured_data.reshape((params['sequence_length'],input_length))
            elif model == 'rnn_2':
                tmp = []
                for j in range(num_filters):
                    tmp.append(featured_data[::,j*num_channels:j*num_channels+num_channels].reshape((params['sequence_length'],input_length/num_filters)))
                featured_data = np.array(tmp)
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
    #if model == 'rnn_2':
    #    data_input = (np.array(data_input)).transpose((1,0,2,3))
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

def reduce_dimension(data,num_components,reducer):
    if reducer == 'ICA':
        ica =  FastICA(n_components=num_components)
        new_features = ica.fit_transform(data)
    elif reducer == 'PCA':
        pca = PCA(n_components=num_components)
        new_features =  pca.fit_transform(data)
    return new_features

def remove_symmetry(data,active_channels,num_channels,combiner):

    max_channels = len(active_channels)
    data = np.array(data,dtype=object)

    for i in range(max_channels/2):
        if active_channels[i] and active_channels[max_channels-i-1]:
            if combiner=='ratio':
                data[::,1+i] = np.asarray(data[::,1+i],np.float)/np.asarray(data[::,max_channels-i],np.float)
            elif combiner == 'diff':
                tm =  np.asarray(data[::,1+i],np.float)-np.asarray(data[::,max_channels-i],np.float)
                data[::,1+i] = tm
            elif combiner == 'max':
                data[::,1+i] = np.max((data[::,1+i],data[::,max_channels-i]),axis=0)
            elif combiner == 'mul':
                data[::,1+i] = np.asarray(data[::,1+i],np.float)*np.asarray(data[::,max_channels-i],np.float)
        elif active_channels[max_channels-i-1]:
            data[::,1+i] = data[::,max_channels-i]

    data =  data[::,0:(max_channels/2+1)]
    return data[::,(0,1,2,3,4,6,7)]

def get_fully_processed_data(params,other_params,model):
    filters = other_params['filters']
    train_data,val_data,test_data=None,None,None

    num_channels = params['num_channels']
    normalize = other_params['normalize']

    ignore_symmetrical_correlation = other_params['ignore_symmetrical_correlation']
    active_channels = other_params['active_channels']
    merger = other_params['merger']
    reducer = other_params['reducer']
    num_components = other_params['num_components']
    balance_data = other_params['balance_data']
    reduce_dimensionality = other_params['reduce_dimensionality']
    reduce_dimensionality_before_reshape = other_params['reduce_dimensionality_before_reshape']
    
    for filter_name,apply_ in filters.iteritems():
        if apply_:
            train_data_, val_data_, test_data_ = load_data(active_channels,ignore_symmetrical_correlation,filter_name=filter_name)

            if balance_data:
                train_data_ = get_balanced_data(train_data_)
                val_data_ = get_balanced_data(val_data_)
                test_data_ =  get_balanced_data(test_data_)

            if not(ignore_symmetrical_correlation):
                #print 'train+ ',np.array(train_data_).shape
                print np.array(train_data_)[0:2]
                train_data_ =  remove_symmetry(train_data_,active_channels, num_channels, other_params['electrode_combiner'])
                #print 'train++ ',train_data_.shape
                print train_data_[0:2]
                val_data_ = remove_symmetry(val_data_,active_channels, num_channels, other_params['electrode_combiner'])
                test_data_ =  remove_symmetry(test_data_,active_channels, num_channels, other_params['electrode_combiner'])

            if train_data is None:
                train_data = np.array(train_data_,dtype=object)
                val_data = np.array(val_data_,dtype=object)
                test_data = np.array(test_data_,dtype=object)
            else:
                if merger == 'concat':
                    train_data = np.concatenate((train_data,np.array(train_data_,dtype=object)[::,1::]),axis=1)
                    val_data = np.concatenate((val_data,np.array(val_data_,dtype=object)[::,1::]),axis=1)
                    test_data = np.concatenate((test_data,np.array(test_data_,dtype=object)[::,1::]),axis=1)
                elif merger == 'avg':
                    train_data[::,1::] =  np.add(np.asarray(train_data[::,1::],np.float),np.asarray(np.array(train_data_)[::,1::],float))
                    val_data[::,1::] =  np.add(np.asarray(val_data[::,1::],np.float),np.asarray(np.array(val_data_)[::,1::],float))
                    test_data[::,1::] =  np.add(np.asarray(test_data[::,1::],np.float),np.asarray(np.array(test_data_)[::,1::],float))

    if merger=='avg':
        train_data[::,1::] = np.true_divide(np.asarray(train_data[::,1::],np.float),int(params['num_filters']))
        val_data[::,1::]= np.true_divide(np.asarray(val_data[::,1::],np.float),int(params['num_filters']))
        test_data[::,1::]= np.true_divide(np.asarray(test_data[::,1::],np.float),int(params['num_filters']))
        #train_data+=val_data

        #print np.array(train_data).shape
        #print train_data[0],train_data[1]
    #print train_data.shape
    if reduce_dimensionality and reduce_dimensionality_before_reshape:
        tmp = train_data[:,0]
        tmp = (np.reshape(tmp,(len(tmp),1)))
        print tmp
        train_data = reduce_dimension(train_data[::,1::],num_components,reducer)
        train_data = np.concatenate((tmp,train_data),axis=1)
        print train_data.shape
        
        tmp = val_data[:,0]
        tmp = (np.reshape(tmp,(len(tmp),1)))
        val_data = reduce_dimension(val_data[::,1::],num_components,reducer)
        val_data = np.concatenate((tmp,val_data),axis=1)
        
        tmp = test_data[:,0]
        tmp = (np.reshape(tmp,(len(tmp),1)))
        test_data = reduce_dimension(test_data[::,1::],num_components,reducer)
        test_data = np.concatenate((tmp,test_data),axis=1)

    train_data = train_data.tolist()
    val_data = val_data.tolist()
    test_data = test_data.tolist()

    train_input, train_output = reformat_data(train_data,params,model=model)
    params['ignore_3rd_class'] = False
    val_input, val_output = reformat_data(val_data,params,model=model)
    test_input, test_output = reformat_data(test_data,params,model=model)


    if reduce_dimensionality and not(reduce_dimensionality_before_reshape):
        # train_input = reduce_dimension(train_input,num_components*params['time_laps']*params['sampling_rate'],reducer)
        # val_input = reduce_dimension(val_input,num_components*params['time_laps']*params['sampling_rate'],reducer)
        # test_input = reduce_dimension(test_input,num_components*params['time_laps']*params['sampling_rate'],reducer)
        nd = other_params['new_dim']
        print '\ndim_reduction',np.array(train_input).shape
        train_input = reduce_dimension(train_input,nd,reducer)
        print np.array(train_input).shape
        val_input = reduce_dimension(val_input,nd,reducer)
        test_input = reduce_dimension(test_input,nd,reducer)

    c1 = zip(train_input,train_output)
    random.shuffle(c1)
    train_input,train_output=  zip(*c1)

    c2 = zip(val_input,val_output)
    random.shuffle(c2)
    val_input,val_output=  zip(*c2)

    c3 = zip(test_input,test_output)
    random.shuffle(c3)
    test_input,test_output=  zip(*c3)

    train_input = np.array(train_input,dtype=object)
    test_input = np.array(test_input,dtype=object)
    val_input = np.array(val_input,dtype=object)

    if model == 'rnn_2':
        train_input = train_input.transpose((1,0,2,3))
        test_input = test_input.transpose((1,0,2,3))
        val_input = val_input.transpose((1,0,2,3))

    if normalize:    
        scaler = preprocessing.StandardScaler().fit(train_input)
        train_input = scaler.transform(train_input)
        test_input = scaler.transform(test_input)
        val_input = scaler.transform(val_input)
    return (train_input,train_output,val_input,val_output,test_input,test_output)

def print_dict(dict):
    for key, val in dict.iteritems():
        print key,': ',val
def main(fil=[], merger='concat' ,time_laps=4):
    all_models = ['rnn_1','rnn_2','rfc','rnn_rfc','svm']
    model = all_models[-1]

    train = bool(0) ## req. only for rnn to switch b/w train and test
    test = bool(1) ## to switch testing b/w test and val

    merger = merger

    balance_data = True
    
    reducer = 'ICA'
    reduce_dimensionality = False
    reduce_dimensionality_before_reshape = False
    new_dim = 500
    
    ignore_3rd_class = False
    
    ignore_symmetrical_correlation = True   #to ignore symmetrical correlation among electrodes
    electrode_combiner = 'mul'   # used if ignore_symmetrical_correlation is False
    normalize = False
    
    sequence_length = 64
    num_channels = 14
    time_laps = time_laps
    sampling_rate = 1
    sampling_types = {'avg':False, 'rms': False, 'max': True, 'mode':False}
    model_specific_params ={}
    toprint = ""

    ## 'AF3',  'F7',  'F3',  'FC5',  'T7',  'P7',  'O1',  'O2',  'P8',  'T8',  'FC6',  'F4',  'F8',  'AF4' 

    dict_active_channels = {'front_left': [1,1,1,1,0,0,0,0,0,0,0,0,0,0],
                            'front_right':[0,0,0,0,0,0,0,0,0,0,1,0,1,1],
                            'back_left':  [0,0,0,0,0,1,1,0,0,0,0,0,0,0],
                            'back_right': [0,0,0,0,0,0,0,1,1,0,0,0,0,0],
                            'front':      [1,1,1,1,0,0,0,0,0,0,1,0,1,1],
                            'back':       [0,0,0,0,0,1,1,1,1,0,0,0,0,0],
                            'all':        [1,1,1,1,0,1,1,1,1,0,1,0,1,1]
                            
    }
    filters = {'':True,'alpha':True,'beta':True,'theta':True,'delta':True,'gamma':True}

    #for combiner in ['diff','ratio','max','mul']:
	#for c in range(14):
	#for pos,val in dict_active_channels.iteritems():    
    #for fltrs in ['110000','101000','100100','100010','100001','011000','010100','010010','010001','001100','001010','001001','000110','000101','000011']:
    #for fil in ['110111']:
    #for i in range(1,64):
    #fil = list('{0:6b}'.format(i))
    # for redr in ['PCA','ICA']:
    #for stype in ['avg','rms','max','avg.rms','avg.max','rms.max','avg.rms.max']:
    #for stype in ['rms']:
    #for sr in [1]:
    print '\n\n\n'
    toprint = ''
    # sequence_length = sr*time_laps
    # toprint = 'sequence_length: '+str(sequence_length)
    # print '\nSequence_Length:, ',sequence_length
    #for min_samples_split_ in [2,4,6,8,10,15,20,40,60,100]:
    # reducer = redr
    # print '\nReducer:',redr
    # for ndim in [25]: 
    # #    print '\n'+pos
    #sampling_type = stype
    # for sty in stype.split('.'):
    #     toprint+='\t'+sty 
    #sampling_rate = sr
    # time_laps = tl
    # # new_dim = ndim
    # # print 'new_dim', new_dim
    # print '\nfilter: ',fl
    #print '\nsampling_types: ',toprint
    #print '\nSampling_rate: ',sampling_rate
    #print '\nMin_Sample_Split: ',min_samples_split_
    #toprint +='\t'+str(sampling_rate)+' '+str(sampling_rate)+' '
    #print '\nTime_Laps: ',time_laps
    active_channels = dict_active_channels['all']
    #electrode_combiner = combiner
    #toprint += '\t'+'Electrode Combiner: '+combiner
    #active_channels = [0]*14
    #active_channels[c] = 1;
    #toprint+='\t'+'channel: '+str(c+1)

    #active_channels = [1,0,0,1,0,1,1,0,0,0,1,0,0,0]
    # filters = {'':False,'alpha':False,'beta':False,'theta':False,'delta': False,'gamma':False}
    # #filters[fl] = True
    # sampling_types = {'avg':False, 'rms': False, 'max': False, 'mode':False}
    # for sty in stype.split('.'):
    #     sampling_types[sty] = True
    # print '\n'+'Time_laps: ',time_laps
    # print 'Merger: ',merger
    # st = 'Filters: '
    # for i in range(6):
    #     if fil[i]=='1':
    #         if i==0:
    #             filters[''] = True
    #             st+='none, '
    #         elif i==1:
    #             filters['alpha'] = True
    #             st+='alpha, '
    #         elif i==2:
    #             filters['beta'] =True
    #             st+='beta, '
    #         elif i==3:
    #             filters['theta'] =True
    #             st+='theta, '
    #         elif i==4:
    #             filters['delta'] =True
    #             st+='delta, '
    #         else:
    #             filters['gamma'] = True
    #             st+= 'gamma, '
    # print '\n'+st
    # toprint +='\t'+st

    if model == 'rnn_1':
        model_specific_params = {'epoch':1000,'batch_size':72, 'num_hidden':32}
    elif model == 'rnn_2':
        merger = 'concat'
        model_specific_params = {'epoch':200,'batch_size':72, 'num_hidden_lstm':32, 'num_hidden_fcn':32}
    elif model == 'rnn_rfc':
        merger = 'concat'
        model_specific_params = {'epoch':200,'batch_size':72, 'num_hidden_lstm':32, 'num_hidden_fcn':32,'n_estimators': 180,'min_samples_split':10}
    elif model == 'rfc':
        model_specific_params = {'n_estimators': 180,'min_samples_split':10}
    elif model == 'svm':
        model_specific_params = {}

    num_channels = sum(active_channels)
    if not(ignore_symmetrical_correlation):
        num_channels/=2
        num_channels+=1

    num_filters = 0

    for key,val in filters.iteritems():
        if val:
            num_filters+=1

    num_sampling_types = 0
    for key,val in sampling_types.iteritems():
        if val:
            num_sampling_types+=1

    num_components = 3*num_filters
    num_cols = num_channels*num_filters
    if merger=='avg':
        num_cols = num_channels

    if reduce_dimensionality_before_reshape and reduce_dimensionality:
        num_cols = num_components

    params = {'num_sampling_types':num_sampling_types,'num_channels':num_channels,'num_cols':num_cols,'test':test,'ignore_3rd_class':ignore_3rd_class,
    'num_filters':num_filters,'time_laps':time_laps,'sampling_rate':sampling_rate, 'sampling_types':sampling_types,
    'sequence_length':sequence_length,'model_specific_params': model_specific_params, 'toprint': toprint}

    other_params = {'num_components':num_components,'reduce_dimensionality':reduce_dimensionality,'electrode_combiner': electrode_combiner,
    'balance_data':balance_data, 'reducer': reducer, 'merger':merger, 'ignore_symmetrical_correlation':ignore_symmetrical_correlation,'normalize':normalize,
    'active_channels':active_channels,'reduce_dimensionality_before_reshape':reduce_dimensionality_before_reshape, 'filters':filters, 'new_dim':new_dim}
    #try:
    print toprint+'\n'
    print_dict(params)
    print '\n'
    print_dict(other_params)

    if model=='rnn_rfc':
        model_ = 'rnn_2'
        data1 = get_fully_processed_data(params,other_params,model_)
        model_ = 'rfc'
        params['sampling_rate'] = 1
        data2 = get_fully_processed_data(params,other_params,model_)
        params['sampling_rate'] = 16
    else:
        data = get_fully_processed_data(params,other_params,model)
    #except:
    #    print '\nError Precessing Data'
    #    #continue
    #print np.array(train_input).shape
    # print np.array(train_output).shape
    #print np.array(test_input).shape
    # print np.array(test_output).shape
    if model == 'rnn_1':
        rnn_1.main(params,data,train=train)
    elif model == 'rnn_2':
        rnn_2.main(params,data,train= train)
    elif model == 'svm':
        models.SVM(params,data)
    elif model == 'rfc':
        models.RFC(params,data)
    elif model == 'rnn_rfc':
        rnn_rfc.main(params,data1,data2,train = train)


if __name__ == '__main__':
    #for i in [3.0,1.5,1,0.5,0.25,0.125]:
    #    print 'time-laps = ',i,' ',
    #    main(i)
    main()

    # for time_laps in [3.0,1.5,1,0.5,0.25,0.125]:
    #     for i in range(1,16):
    #         fil = list('{0:4b}'.format(i))
    #         main(fil,'concat',time_laps)
    #         main(fil,'avg',time_laps)



