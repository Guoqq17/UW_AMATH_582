close all; clear; clc;

%% import data and set up
% v=VideoReader('ski_drop_low.mp4');
v=VideoReader('monte_carlo_low.mp4');

% convert data to matrix format
width=v.Width;
height=v.Height;
images=zeros(width*height, v.NumberOfFrames);
for i=1:1:v.NumberOfFrames
    cur_img_temp=rgb2gray(read(v,i));
%     cur_img=reshape(cur_img_temp(:,233:728),width*height,1);
    cur_img=reshape(cur_img_temp,width*height,1);
    images(:,i)=double(cur_img);
end

% create time series
t_temp=linspace(0,v.CurrentTime, v.NumberOfFrames+1);
t=t_temp(1:end-1);
dt=t(2)-t(1);

%% DMD
% SVD
X1=images(:,1:end-1);
X2=images(:,2:end);
[U,Sigma,V]=svd(X1,'econ');
figure(1)
subplot(1,2,1)
plot(diag(Sigma)/sum(diag(Sigma)), '-o')
xlabel('Order of singular value','Fontsize',12)
ylabel('Proportion','Fontsize',12)

subplot(1,2,2)
plot(diag(Sigma(1:100, 1:100))/sum(diag(Sigma)), '-o')
xlabel('Order of singular value','Fontsize',12)
ylabel('Proportion','Fontsize',12)

temp=diag(Sigma)/sum(diag(Sigma));
acc_p=zeros(1,length(temp));
for i=1:1:length(temp)
    acc_p(i)=sum(temp(1:i));
end

figure(2)
plot(acc_p, '-o')
xlabel('Order of singular value','Fontsize',12)
ylabel('Total proportion','Fontsize',12)
ylim([0,1])
% low-rank approximation
r=100; % define the rank based on the singular value spectrum
U2=U(:,1:r);
Sigma2=Sigma(1:r,1:r);
V2=V(:,1:r);
S=U2'*X2*V2/Sigma2; % low rank
[eV,D]=eig(S);
% Phi=X2*V2/Sigma2*eV; % DMD modes
Phi=U2*eV; % DMD modes

lambda=diag(D);
omega=log(lambda)/dt;
bg = find(abs(omega)<1e-2);
omega_bg = omega(bg); % background eigenvalues
Phi_bg = Phi(:,bg); % DMD background mode

% reconstruct low-rank DMD
x1=X1(:,1);
y0=Phi_bg\x1;
x_modes=zeros(length(omega_bg),length(t));
for iter=1:length(t)
    x_modes(:,iter)=(y0.*exp(omega_bg*t(iter)));
end
X_dmd=Phi_bg*x_modes;

%% sparse method 1
% X_sparse=images-abs(X_dmd);
% R=X_sparse.*(X_sparse<0);
% X_dmd=R+abs(X_dmd);
% X_s_dmd=X_sparse-R;

%% sparse method 2
X_sparse=images-abs(X_dmd);
X_s_dmd=X_sparse + 150;
%% show results
% for report (3 snapshots)
snaps=[100,200,300];
figure(3)
for i=1:1:length(snaps)
    subplot(3,3,(i-1)*3+1)
    img1=reshape(uint8(images(:,snaps(i))),height,width);
    imshow(img1)
    title(['Original video, Frame ', num2str(snaps(i))])
    subplot(3,3,(i-1)*3+2)
    img2=reshape(uint8(X_dmd(:,snaps(i))),height,width);
    imshow(img2)
    title(['Backgroud, Frame ',num2str(snaps(i))])
    subplot(3,3,(i-1)*3+3)
    img3=reshape(uint8(X_s_dmd(:,snaps(i))),height,width);
    imshow(img3)
    title(['Foregroud, Frame ', num2str(snaps(i))])
end

% for viewing (all snapshots)
% figure(4)
% for ind_show=1:v.numberOfFrames
%     subplot(1,3,1)
%     img1=reshape(uint8(images(:,ind_show)),height,width);
%     imshow(img1)
%     title('Original video')
%     subplot(1,3,2)
%     img2=reshape(uint8(X_dmd(:,ind_show)),height,width);
%     imshow(img2)
%     title('Backgroud extracted by Low-rank DMD')
%     subplot(1,3,3)
%     img3=reshape(uint8(X_s_dmd(:,ind_show)),height,width);
%     imshow(histeq(img3))
%     title('Foregroud extracted by sparse DMD')
%     drawnow
% end