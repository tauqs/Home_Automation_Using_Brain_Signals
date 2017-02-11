function p = CallFunc
clc;
clear all;

files = dir('*csv')
files.name
for k=1:length(files)
    EEG_signal_processing(files(k).name)
end
%EEG_signal_processing('Trial.csv')


