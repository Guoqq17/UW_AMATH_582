close all; clear; clc;

%% import data
v=VideoReader('ski_drop_low.mp4');
% v=VideoReader('monte_carlo_low.mp4');

%% convert data to MATLAB format
width=v.Width;
height=v.Height;
width=728-232;
images=zeros(width*height, v.NumberOfFrames);
for i=1:1:v.NumberOfFrames
    cur_img_temp=rgb2gray(read(v,i));
    cur_img=reshape(cur_img_temp(:,233:728),width*height,1);
    images(:,i)=double(cur_img);
end
t_temp=linspace(0,v.CurrentTime, v.NumberOfFrames+1);
t=t_temp(1:end-1);
dt=t(2)-t(1);

%% DMD
% SVD
X1=images(:,1:end-1);
X2=images(:,2:end);
[U,Sigma,V]=svd(X1,'econ');
figure(1)
plot(diag(Sigma)/sum(diag(Sigma)), '-o')
xlabel('Order of singular value','Fontsize',12)
ylabel('Proportion','Fontsize',12)

% low-rank approximation
r=1; % define the rank based on the singular value spectrum
U2=U(:,1:r);
Sigma2=Sigma(1:r,1:r);
V2=V(:,1:r);
S=U2'*X2*V2*diag(1./diag(Sigma2)); % low rank
[eV,D]=eig(S);
Phi=X2*V2/Sigma2*eV; % DMD modes

% reconstruct low-rank DMD
lambda=diag(D);
omega=log(lambda)/dt;
x1=X1(:,1);
y0=Phi\x1;

x_modes=zeros(r,length(t));
for iter=1:length(t)
    x_modes(:,iter)=(y0.*exp(omega*t(iter)));
end

X_dmd=Phi*x_modes;

%% sparse
X_sparse=images-abs(X_dmd);
R=X_sparse.*(X_sparse<0);
X_dmd=R+abs(X_dmd);
X_s_dmd=X_sparse-R;
%%
% ind_show=200;
figure(2)
for ind_show=1:v.numberOfFrames
    subplot(1,3,1)
    img1=reshape(uint8(images(:,ind_show)),height,width);
    imshow(img1)
    title('Original video')
    subplot(1,3,2)
    img2=reshape(uint8(X_dmd(:,ind_show)),height,width);
    imshow(img2)
    title('Backgroud extracted by Low-rank DMD')
    subplot(1,3,3)
    img3=reshape(uint8(X_s_dmd(:,ind_show)),height,width);
    imshow(histeq(img3))
    title('Foregroud extracted by sparse DMD')
    drawnow
end









