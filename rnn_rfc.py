import numpy as np
import random
from random import shuffle
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import os
import csv
#import data_processing_util as dlu

def build_multi_channel_lstm_network(params):

    num_channels = params['num_channels']
    time_laps = params['time_laps']
    sequence_length = params['sequence_length']
    sampling_rate = params['sampling_rate']
    num_cols = params['num_cols']
    num_filters= params['num_filters']
    num_hidden_lstm = params['model_specific_params']['num_hidden_lstm']
    num_hidden_fcn = params['model_specific_params']['num_hidden_fcn']

    input_length = int(num_channels*time_laps*sampling_rate/float(sequence_length))

    #data = [0]*num_filters
    cell = [0]*num_filters
    val = [0]*num_filters
    last = [0]*num_filters

    #data = tf.convert_to_tensor(data)
    #cell = tf.convert_to_tensor(cell)
    #val = tf.convert_to_tensor(val)
    data = tf.placeholder(tf.float32, [None,None, sequence_length,input_length]) #Number of examples, number of input, dimension of each input
    print 'data: ',data.get_shape()
    target = tf.placeholder(tf.float32, [None, 3])
    for i in range(num_filters):
        
        with tf.variable_scope('lstm_'+str(i)): 
            cell[i] = tf.nn.rnn_cell.LSTMCell(num_hidden_lstm,state_is_tuple=True)
            val[i], _ = tf.nn.dynamic_rnn(cell[i], data[i], dtype=tf.float32)

        val[i] = tf.transpose(val[i], [1, 0, 2])
        last[i] = tf.gather(val[i], int(val[i].get_shape()[0]) - 1)

    #data = tf.pack(data)
    #cell = tf.pack(cell)
    #val = tf.pack(val)
    last = tf.pack(last,axis=1)
    print 'last: ',last.get_shape()
    
    dim =  last.get_shape()[1]*last.get_shape()[2]
    last_f = tf.reshape(last,[-1,int(dim)])
    print 'last_f: ',last_f.get_shape()

    weight_f = tf.Variable(tf.truncated_normal([num_hidden_lstm*num_filters, int(target.get_shape()[1])]))
    bias_f = tf.Variable(tf.constant(0.1, shape=[int(target.get_shape()[1])]))
    prediction = tf.nn.softmax(tf.matmul(last_f, weight_f) + bias_f)

    # weight_f = tf.Variable(tf.truncated_normal([num_hidden_lstm*num_filters, num_hidden_fcn]))
    # bias_f = tf.Variable(tf.constant(0.1, shape=[num_hidden_fcn]))
    # hidden = tf.nn.relu(tf.matmul(last_f, weight_f) + bias_f)
    # weight = tf.Variable(tf.truncated_normal([num_hidden_fcn, int(target.get_shape()[1])]))
    # bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
    # prediction = tf.nn.softmax(tf.matmul(hidden, weight) + bias)
    
    cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)

    mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    zero_int64 = tf.cast(0,tf.int64)
    t1_1 = tf.equal(zero_int64,tf.argmax(prediction, 1))
    t1_2 = tf.equal(zero_int64,tf.argmax(target,1))

    corrects_class_1 = tf.mul(tf.cast(t1_1,tf.float32), tf.cast(t1_2,tf.float32))
    len_class_1 = tf.reduce_sum(tf.cast(tf.equal(zero_int64,tf.argmax(target,1)),tf.float32))
    accuracy_class_1 = tf.truediv(tf.reduce_sum(corrects_class_1),len_class_1)
    
    one_int64 = tf.cast(1,tf.int64)
    t2_1 = tf.equal(one_int64,tf.argmax(prediction, 1))
    t2_2 = tf.equal(one_int64,tf.argmax(target,1))

    corrects_class_2 = tf.mul(tf.cast(t2_1,tf.float32), tf.cast(t2_2,tf.float32))
    len_class_2 = tf.reduce_sum(tf.cast(tf.equal(one_int64,tf.argmax(target,1)),tf.float32))
    accuracy_class_2 = tf.truediv(tf.reduce_sum(corrects_class_2),len_class_2)
    
    two_int64 = tf.cast(2,tf.int64)
    t3_1 = tf.equal(two_int64,tf.argmax(prediction, 1))
    t3_2 = tf.equal(two_int64,tf.argmax(target,1))

    corrects_class_3 = tf.mul(tf.cast(t3_1,tf.float32), tf.cast(t3_2,tf.float32))
    len_class_3 = tf.reduce_sum(tf.cast(tf.equal(two_int64,tf.argmax(target,1)),tf.float32))
    accuracy_class_3 = tf.truediv(tf.reduce_sum(corrects_class_3),len_class_3)
    

    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)

    return (sess,minimize,error,data,target,accuracy_class_1,accuracy_class_2,accuracy_class_3,last_f)

def run_lstm_model(tensorflow_data,data1,data2,saver,train=True,test=False, batch_size=72, epoch=200):

    sess,minimize,error,data,target,accuracy_class_1,accuracy_class_2,accuracy_class_3,last_f = tensorflow_data
    train_input1,train_output1,val_input1,val_output1,test_input1,test_output1 = data1
    train_input2,train_output2,val_input2,val_output2,test_input2,test_output2 = data2

    train_input1 = np.array(train_input1)
    train_output1  = np.array(train_output1)

    if(not(test)):
        test_input1 = val_input1
        test_output1 = val_output1
        test_input2 = val_input2
        test_output2 = val_output2

    no_of_batches = int(train_input1.shape[1]) / batch_size
    
    if train:
        #saver = tf.train.Saver()        
        #saved_model = None
        for i in range(epoch):
            try:
                ptr = 0
                for j in range(no_of_batches):
                    inp, out = train_input1[::,ptr:ptr+batch_size,::,::], train_output1[ptr:ptr+batch_size]
                    ptr+=batch_size
                    #print inp.shape
                    sess.run(minimize,{data: inp, target: out})
                    #incorrect = sess.run(error,{data: train_input, target: train_output})
                    #print('Epoch {:2d} loss {:3.1f}'.format(i + 1, incorrect))
                #print "Epoch ",str(i)
                incorrect = sess.run(error,{data: train_input1, target: train_output1})
                print('Epoch {:2d} loss {:3.5f}'.format(i + 1, incorrect))
                
                if (i+1)%50 == 0:
                    saved_model = saver.save(sess,'./Saved_Models/lstm_model',global_step = i+1) 
                    incorrect = sess.run(error,{data: test_input1, target: test_output1})

                    acc_class_1 = sess.run(accuracy_class_1,{data: test_input1, target: test_output1})
                    acc_class_2 = sess.run(accuracy_class_2,{data: test_input1, target: test_output1})
                    acc_class_3 = sess.run(accuracy_class_3,{data: test_input1, target: test_output1})

                    print('Accuracy {:3.5f}%'.format(100-100 * incorrect))
                    print('Accuracy off_to_on {:3.4f}%'.format(100 * acc_class_1))
                    print('Accuracy on_to_off {:3.4f}%'.format(100 * acc_class_2))
                    print('Accuracy do_nothing {:3.4f}%'.format(100 * acc_class_3)) 

                    with open('rnn2_stat.txt','a') as fp:
                        fp.write('Epoch: '+str(i+1)+'Accuracy {:3.5f}%'.format(100-100 * incorrect)+'\n')
                        fp.close()
            except KeyboardInterrupt:
                print 'exception'
                saved_model = saver.save(sess,'./Saved_Models/lstm_model',global_step = i+1)

                incorrect = sess.run(error,{data: test_input1, target: test_output1})

                acc_class_1 = sess.run(accuracy_class_1,{data: test_input1, target: test_output1})
                acc_class_2 = sess.run(accuracy_class_2,{data: test_input1, target: test_output1})
                acc_class_3 = sess.run(accuracy_class_3,{data: test_input1, target: test_output1})

                print('Accuracy {:3.5f}%'.format(100-100 * incorrect))
                print('Accuracy off_to_on {:3.4f}%'.format(100 * acc_class_1))
                print('Accuracy on_to_off {:3.4f}%'.format(100 * acc_class_2))
                print('Accuracy do_nothing {:3.4f}%'.format(100 * acc_class_3))
                #sys.exit()
        saved_model = saver.save(sess,'./Saved_Models/lstm_model',global_step = i+1) 
        incorrect = sess.run(error,{data: test_input1, target: test_output1})

        acc_class_1 = sess.run(accuracy_class_1,{data: test_input1, target: test_output1})
        acc_class_2 = sess.run(accuracy_class_2,{data: test_input1, target: test_output1})
        acc_class_3 = sess.run(accuracy_class_3,{data: test_input1, target: test_output1})

        print('Accuracy {:3.5f}%'.format(100-100 * incorrect))
        print('Accuracy off_to_on {:3.4f}%'.format(100 * acc_class_1))
        print('Accuracy on_to_off {:3.4f}%'.format(100 * acc_class_2))
        print('Accuracy do_nothing {:3.4f}%'.format(100 * acc_class_3)) 

        with open('rnn2_stat.txt','a') as fp:
            fp.write('Epoch: '+str(i+1)+'Accuracy {:3.5f}%'.format(100-100 * incorrect)+'\n')
            fp.close()
    else:
        ckpt = tf.train.get_checkpoint_state('./Saved_Models')
        if ckpt and ckpt.model_checkpoint_path:
            print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print '\nError!!! first train the model, then test it\n'
        
        incorrect = sess.run(error,{data: test_input, target: test_output})

        acc_class_1 = sess.run(accuracy_class_1,{data: test_input, target: test_output})
        acc_class_2 = sess.run(accuracy_class_2,{data: test_input, target: test_output})
        acc_class_3 = sess.run(accuracy_class_3,{data: test_input, target: test_output})

        print('Accuracy {:3.5f}%'.format(100-100 * incorrect))
        print('Accuracy off_to_on {:3.4f}%'.format(100 * acc_class_1))
        print('Accuracy on_to_off {:3.4f}%'.format(100 * acc_class_2))
        print('Accuracy do_nothing {:3.4f}%'.format(100 * acc_class_3))

def train_rfc(tensorflow_data,params,data1,data2,saver):

    sess,minimize,error,data,target,accuracy_class_1,accuracy_class_2,accuracy_class_3,last_f = tensorflow_data
    train_input1,train_output1,val_input1,val_output1,test_input1,test_output1 = data1
    train_input2,train_output2,val_input2,val_output2,test_input2,test_output2 = data2

    test = params['test']
    if(not(test)):
        test_input1 = val_input1
        test_output1 = val_output1
        test_input2 = val_input2
        test_output2 = val_output2

    ckpt = tf.train.get_checkpoint_state('./Saved_Models')
    if ckpt and ckpt.model_checkpoint_path:
        print ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print '\nError!!! first train the model, then test it\n'
    

    train_features = (sess.run(last_f,{data: train_input1, target: train_output1}))
    test_features = (sess.run(last_f,{data: test_input1, target: test_output1}))

    print train_features.shape

    train_input2 = np.concatenate((train_input2,train_features),axis=1)
    test_input2 = np.concatenate((test_input2,test_features),axis=1)

    print 'Train_input2: ',train_input2.shape
    print 'Test_input2:, ',test_input2.shape

    r =10
    if test:
        r=1
    acc = float(0)

    for i in range(r):
        est = 20*(i+1)
        #est = params['model_specific_params']['n_estimators']
        if test:
            est = params['model_specific_params']['n_estimators']
        model = RandomForestClassifier(n_estimators=est)
        #model = RandomForestClassifier(n_estimators=est,min_samples_split=100,class_weight={0:4,1:4,2:2})
        model = model.fit(train_input2,train_output2)
        print 'n-estimators: ',est,
        
        #if test:
        accuracy = 100*model.score(test_input2,test_output2)
        #else:
        #    accuracy = 100*model.score(val_input2,val_output2)
            
        acc+=accuracy
        print  ' Accuracy: ',accuracy,'%'
    avgAcc=  'Avg accuracy: '+str(acc/float(r))+'%'
    print avgAcc
    with open('stat.txt','a') as fp:
        fp.write(params['toprint']+' '+avgAcc+'\n')
        fp.close()


def main(params,data1,data2,train=True):

    #train_input,train_output,val_input,val_output,test_input,test_output = dlu.main()
    train_input1,train_output1,val_input1,val_output1,test_input1,test_output1 = data1
    train_input2,train_output2,val_input2,val_output2,test_input2,test_output2 = data2
    #train_input+=val_input
    #train_output+=val_output

    
    print '\n\nData1:'
    print 'train_data shape: ',np.array(train_input1).shape,np.array(train_output1).shape
    print 'val_data shape: ',np.array(val_input1).shape,np.array(val_output1).shape
    print 'test_data shape: ',np.array(test_input1).shape,np.array(test_output1).shape

    print '\nData2:'
    print 'train_data shape: ',np.array(train_input2).shape,np.array(train_output2).shape
    print 'val_data shape: ',np.array(val_input2).shape,np.array(val_output2).shape
    print 'test_data shape: ',np.array(test_input2).shape,np.array(test_output2).shape

    #np.reshape(train_input,())

    tensorflow_data =  build_multi_channel_lstm_network(params = params)
    saver = tf.train.Saver()

    batch_size = params['model_specific_params']['batch_size']
    epoch = params['model_specific_params']['epoch']

    with open('rnn2_stat.txt','a') as fp:
        fp.write('\n\n'+params['toprint']+'\n')
        fp.close()

    if train:
        run_lstm_model(tensorflow_data,data1,data2,saver,test =params['test'], batch_size=batch_size,epoch=epoch)
    #run_lstm_model(tensorflow_data,data1,data2,saver,train = False,test = params['test'])
    train_rfc(tensorflow_data,params,data1,data2,saver)

if __name__ == '__main__':
    main()
