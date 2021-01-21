% Clean workspace
clear all; close all; clc

% load data
currentPath=fileparts(mfilename('fullpath'));
load('subdata.mat');

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

% visulize the data
for j = 1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    M = max(abs(Un),[],'all');
    close all, isosurface(X,Y,Z,abs(Un)/M, 0.7)
    axis([-20 20 -20 20 -20 20]), grid on, drawnow
    pause(1)
end

%% 1. average the spectrum to determine the frequency signature generate by the submarine
ave = zeros(n,n,n);
for j = 1:49
    % reshape
    un(:,:,:)=reshape(subdata(:,j),n,n,n);
    % fft
    utn = fftn(un);
    % sum frequency data
    ave = ave + utn;
end
% average frequency data
ave = abs(fftshift(ave))/49;
% normalize
ave = ave/max(max(max(ave)));


[val, ind] = max(ave(:));
[a,b,c] = ind2sub(size(ave),ind);

% the central frequency
c_x = Kx(a,b,c);
c_y = Ky(a,b,c);
c_z = Kz(a,b,c);


%% 2.filter the data around the center frequency and determine the path of the submarine
filter = exp(-0.2 * ((Kx - c_x).^2 + (Ky - c_y).^2 + (Kz - c_z).^2));
pos_sub = zeros(3, 49);
for j = 1:49
    un(:,:,:)=reshape(subdata(:,j),n,n,n);
    utn = fftn(un);
    utnf = fftshift(utn) .* filter;
    unf = ifftn(utnf);
    
    [val, ind] = max(abs(unf(:)));
    [a,b,c] = ind2sub(size(unf),ind);
    pos_sub(1,j) = a;
    pos_sub(2,j) = b;
    pos_sub(3,j) = c;
end
figure(3)
plot3(x(pos_sub(1,:)), y(pos_sub(2,:)), z(pos_sub(3,:)), '-o')
grid on
xlabel('x')
ylabel('y')
zlabel('z')

%% predict the trajectory

x_pre = x(pos_sub(1, 30: 49));
y_pre = y(pos_sub(2, 30: 49));
p_pre = polyfit(x_pre, y_pre, 3);

x_new = linspace(0, 6, 60);
y_new = polyval(p_pre, x_new);
figure(4)
plot(x(pos_sub(1,:)), y(pos_sub(2,:)), '-o')
hold on
plot(x_pre, y_pre, 'g-o')
plot(x_new, y_new)
legend('original points', 'fitting points', 'predicted trajectory')
grid on
