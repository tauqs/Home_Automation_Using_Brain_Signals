function p = CallFunc
clc;
clear all;

files = dir('*csv')
files.name
for k=1:length(files)
    digital_filter_util(files(k).name)
end
%EEG_signal_processing('Trial.csv')


