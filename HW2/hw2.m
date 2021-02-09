%% understand the data
figure(1)
[y, Fs] = audioread('Floyd.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Sweet Child O Mine');
p8 = audioplayer(y,Fs); playblocking(p8);

%% task 1, GNR
[y_all, Fs] = audioread('GNR.m4a');     % read the audio data
[y, Fs] = audioread('GNR.m4a', [1, 3*Fs]);
tr_gnr = length(y)/Fs;              % record time in seconds
L = tr_gnr;                         % time domin
n = length(y);                      % Fourier modes
t1 = linspace(0, L, n + 1);
t = t1(1:n);
k = (2*pi/L)*[0:n/2-1, -n/2:-1];
ks = fftshift(k);

% create Gabor filter
width_t_all = [10, 100, 1000, 10000];        % width of the filter
width_q_all = [0.02, 0.002, 0.0002, 0.00002];           % width of the filter in frequency domain
num_gabor = 100;                            % number of the time points to take
t_gabor = linspace(0, t(end), num_gabor);   % discretize the time
s_gabor = zeros(length(t_gabor), n);        % matrix to store the Gabor transforms

%% tune wt
figure(2)
for w_t = 1:1:length(width_t_all)
    for i=1:length(t_gabor)
        gabor = exp(-width_t_all(w_t)*(t - t_gabor(i)).^2);
        gyt = fft(gabor.*y.');
        gyts = abs(fftshift(gyt));
        s_gabor(i,:) = gyts;
    end

    % plot the spectrogram
    subplot(1, length(width_t_all), w_t)
    pcolor(t_gabor, ks/(2*pi), log(s_gabor.' + 1)), shading interp
    colormap('hot'), xlabel('Time [sec]'), ylabel('Frequency [Hz]')
    axis([0, tr_gnr, 0, 1000])
    title(['wt = ', num2str(width_t_all(w_t))])
end
%% tune wq
figure(3)
for w_q = 1:1:length(width_q_all)
    for i=1:length(t_gabor)
        gabor = exp(-100*(t - t_gabor(i)).^2);
        gyt = fft(gabor.*y.');
        gyts = abs(fftshift(gyt));
        [val, ind] = max(gyts(n/2:end));
        [a,b] = ind2sub(size(gyts),ind+n/2-1);
        s_filter = exp(-width_q_all(w_q) * ((ks - ks(b)).^2));
        gytf = fftshift(gyt).*s_filter;
        s_gabor(i,:) = abs(gytf);
    end

    % plot the spectrogram
    subplot(1, length(width_q_all), w_q)
    pcolor(t_gabor, ks/(2*pi), log(s_gabor.' + 1)), shading interp
    colormap('hot'), xlabel('Time [sec]'), ylabel('Frequency [Hz]')
    axis([0, tr_gnr, 0, 2000])
    title(['wq = ', num2str(width_q_all(w_q))])
end


%% task 1, Floyd
clear; clc;

[y_all, Fs] = audioread('Floyd.m4a');
step = 6;                               % time step to cut the whole audio to slices
num_gabor = 100;                        % number of the time points to take at each slice
s_gabor_all = zeros(num_gabor*round(length(y_all)/(Fs*step)), length(y_all));
n_all = length(y_all);
L_all = length(y_all)/Fs;
k_all = (2*pi/L_all)*[0:n_all/2-1, -n_all/2:-1];
ks_all = fftshift(k_all);
%%
figure(3)
for s_i=1:length(y_all)/(Fs*step)
    if s_i*step*Fs < length(y_all)
        [y, Fs] = audioread('Floyd.m4a', [(s_i-1)*step*Fs+1, s_i*step*Fs]);
    else
        [y, Fs] = audioread('Floyd.m4a', [(s_i-1)*step*Fs+1, length(y_all)]);
    end
    
    tr_gnr = length(y)/Fs;                  % record time in seconds
    L = tr_gnr;                             % time domin
    n = length(y);                          % Fourier modes
    t1 = linspace(0, L, n + 1);
    t = t1(1:n);
    k = (2*pi/L)*[0:n/2-1, -n/2:-1];
    ks = fftshift(k);

    % create Gabor filter
    width = 100;                                % width of the filter
    t_gabor = linspace(0, t(end), num_gabor);   % discretize the time
    s_gabor = zeros(length(t_gabor), n);

    % create the spectrogram
    for i=1:length(t_gabor)
        gabor = exp(-width*(t - t_gabor(i)).^2);
        gyt = fft(gabor.*y.');
        gyts = abs(fftshift(gyt));
        [val, ind] = max(gyts(n/2:end));
        [a,b] = ind2sub(size(gyts),ind+n/2-1);
        s_filter = exp(-0.05 * ((ks - ks(b)).^2));
        gytf = fftshift(gyt).*s_filter;
        s_gabor(i,:) = abs(gytf);
    end
    
    s_gabor_all((s_i-1)*num_gabor+1:s_i*num_gabor, (s_i-1)*step*Fs+1:s_i*step*Fs) = s_gabor;
    
    pcolor(t_gabor, ks/(2*pi), log(s_gabor.' + 1)), shading interp
    colormap('hot'), xlabel('Time [sec]'), ylabel('Frequency [Hz]')
    axis([0, tr_gnr, 0, 1000])
    title('Log of the spectrogram of GNR')
    drawnow
end

