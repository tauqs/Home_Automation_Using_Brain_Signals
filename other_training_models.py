import numpy as np
import random
from random import shuffle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import os
import csv
#import data_processing_util as dlu

def RFC(params,train_input,train_output,val_input,val_output,test_input,test_output):
    train_input+=val_input
    train_output+=val_output

    print np.array(train_input).shape
    print np.array(test_input).shape

    model = RandomForestClassifier(n_estimators=params['model_specific_params']['n_estimators'])
    model = model.fit(train_input,train_output)

    print  100*model.score(test_input,test_output)

def SVM(params,train_input,train_output,val_input,val_output,test_input,test_output):
    #print subject_trial_count
    #train_input,train_output,val_input,val_output,test_input,test_output =dlu.main(model='svm')

    clf = svm.SVC(decision_function_shape = 'ovo')
    clf.fit(train_input,train_output)

    dec = clf.predict(test_input)
    print zip(test_output,dec)

    # accuracy = 100*sum(test_input == dec)/float(len(test_output))
    #print 'Accuracy = %x %',accuracy

    print clf.score(train_input,train_output)
    print clf.score(test_input,test_output)

def main():
	pass

if __name__ == '__main__':
    main()

