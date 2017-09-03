import numpy as np
import random
from random import shuffle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import os
import csv
#import data_processing_util as dlu

def RFC(params,data):

    train_input,train_output,val_input,val_output,test_input,test_output = data
    #train_input =  np.concatenate((train_input,val_input))
    #train_output = np.concatenate((train_output,val_output))
    #train_input = train_input.tolist()+val_input.tolist()
    #train_output = train_output.tolist()+ val_output.tolist()
    

    test = params['test']
    #test = True
    
    print '\n\ntrain_data shape: ',np.array(train_input).shape
    print 'test_data shape: ',np.array(test_input).shape
    acc = float(0)

    th = 0.75

    r =10
    if test:
        r=1


    for i in range(r):
        est = 20*(i+1)
        #est = params['model_specific_params']['n_estimators']
        if test:
            est = params['model_specific_params']['n_estimators']
        model = RandomForestClassifier(n_estimators=est)
        #model = RandomForestClassifier(n_estimators=est,min_samples_split=100,class_weight={0:4,1:4,2:2})
        model = model.fit(train_input,train_output)
        print 'n-estimators: ',est,
        
        if test:
            accuracy = 100*model.score(test_input,test_output)
        else:
            accuracy = 100*model.score(val_input,val_output)
            '''

            out_prob = model.predict_proba(val_input)
            print zip(out_prob,val_output)
            accuracy = 0
            for i in range(len(val_input)):
                if out_prob[i][0]>=th:
                    if val_output[i]==0:
                        accuracy+=1
                elif out_prob[i][1]>=th:
                    if val_output[i]==1:
                        accuracy+=1
                else:
                    if val_output[i] ==2:
                        accuracy+=1
            accuracy = accuracy/float(len(val_input))
            '''
        acc+=accuracy
        print  ' Accuracy: ',accuracy,'%'
    avgAcc=  'Avg accuracy: '+str(acc/float(r))+'%'
    print avgAcc
    with open('stat.txt','a') as fp:
        fp.write(params['toprint']+' '+avgAcc+'\n')
        fp.close()


def SVM(params,data):
    #print subject_trial_count
    #train_input,train_output,val_input,val_output,test_input,test_output =dlu.main(model='svm')
    train_input,train_output,val_input,val_output,test_input,test_output = data
    if not(params['test']):
        test_input = val_input
        test_output = val_output
    #clf = svm.SVC(kernel='poly')
    clf = svm.NuSVC()
    #clf = svm.LinearSVC()
    clf.fit(train_input,train_output)

    dec = clf.predict(np.asarray(test_input,np.float))
    print zip(test_output,dec)

    # accuracy = 100*sum(test_input == dec)/float(len(test_output))
    #print 'Accuracy = %x %',accuracy

    print clf.score(np.asarray(train_input,np.float),np.asarray(train_output,np.float))
    print clf.score(np.asarray(test_input,np.float),np.asarray(test_output,np.float))

def main():
	pass

if __name__ == '__main__':
    main()

