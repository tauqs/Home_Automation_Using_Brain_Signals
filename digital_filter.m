function p = digital_filter
clc;
clear all;

addpath('Data/Smoothened_and_Digitally_Filtered/')
files = dir('Data/Smoothened_Data/*csv')
files.name
for k=1:length(files)
    digital_filter_util(files(k).name)
end
%EEG_signal_processing('Trial.csv')


