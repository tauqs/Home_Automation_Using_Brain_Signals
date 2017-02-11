function table = EEG_signal_processing(file)
clc;
fclose all;
file = char(file)
numlines = str2num( perl('countlines.pl', file) )
n = numlines - 1
channels = dlmread(file, ',', [1,3,n,16])
Fs = 128
N = 12288
ts = [1/Fs : 1/Fs : N/Fs]
fprintf('Data Extrated.\n')
%figure; % get a new figure
% Repeat for each channel
name = {'AF3','F7','F3','FC5','T7','P7','01','O2','P8','T8','FC6','F4','F8','AF4'}
freq = {'delta','theta','alpha','beta','gamma'}
signal = channels
Delta_filter = designfilt('lowpassiir','FilterOrder',8,'PassbandFrequency',4,'PassbandRipple',0.2, 'SampleRate',Fs);
%fvtool(Delta_filter); % displays frequency response of Delta filter
fprintf('1 figure generated: Frequency response of 4Hz low-pass filter.\n');
% Band-pass filter to extract Theta signal (4 to 8 Hz)
Theta_filter = designfilt('bandpassiir', 'FilterOrder', 20, 'PassbandFrequency1', 4,'PassbandFrequency2',8,'PassbandRipple',1,'SampleRate', Fs);
%fvtool(Theta_filter); % displays frequency response of Theta filter
fprintf('1 figure generated: Frequency response of 4-8Hz band-pass filter.\n');
% Band-pass filter to extract Alpha signal (8 to 13 Hz)
Alpha_filter = designfilt('bandpassiir', 'FilterOrder', 20, 'PassbandFrequency1', 8,'PassbandFrequency2',13,'PassbandRipple',1,'SampleRate', Fs);
%fvtool(Alpha_filter); % displays frequency response of Alpha filter
fprintf('1 figure generated: Frequency response of 8-13Hz band-pass filter.\n');
% Band-pass filter to extract Beta signal (13 to 30 Hz)
Beta_filter = designfilt('bandpassiir', 'FilterOrder', 20, 'PassbandFrequency1', 13,'PassbandFrequency2',30,'PassbandRipple',1,'SampleRate', Fs);
%fvtool(Beta_filter); % displays frequency response of Beta filter
fprintf('1 figure generated: Frequency response of 13-30Hz band-pass filter.\n');
% Band-pass filter to extract Gamma signal (30 to 100 Hz)
Gamma_filter = designfilt('bandpassiir', 'FilterOrder', 20, 'PassbandFrequency1', 30,'PassbandFrequency2',63,'PassbandRipple',1,'SampleRate', Fs);
%fvtool(Gamma_filter); % displays frequency response of Gamma filter
fprintf('1 figure generated: Frequency response of 30-100Hz band-pass filter.\n');

for i = 1:14
    delta(:,i) = filter(Delta_filter,signal(:,i));
    theta(:,i) = filter(Theta_filter,signal(:,i));
    alpha(:,i) = filter(Alpha_filter,signal(:,i));
    beta(:,i) = filter(Beta_filter,signal(:,i));
	gamma(:,i) = filter(Gamma_filter,signal(:,i));
end
data = {delta,theta,alpha,beta,gamma}
for i = 1:size(freq,2)
    fprintf('%d',i)
    str = strrep(file,'.csv','')
    file_name = char(strcat(str,'_',freq(1,i),'.csv'))
    fid = fopen(file_name, 'w')
    fprintf(fid, '%s,', name{1,1:end-1})
    fprintf(fid, '%s\n', name{1,end})
    fclose(fid)
    dlmwrite(file_name,data(i),'precision',12,'-append')
end
