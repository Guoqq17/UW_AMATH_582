%% visulize data
% implay(vidFrames3_2)
%%
close all
% video 1_2
[a1, b1, c1, d1] = size(vidFrames1_2);
thresh = 50;                            % the search threshold to find the light spot
start_point = [326,311];                % start point to search
x1=zeros(1,d1);
y1=zeros(1,d1);
for i=1:1:d1
    img = rgb2gray(vidFrames1_2(:,:,:,i));
    img(:,1:start_point(1)-thresh) = 0;
    img(:,start_point(1)+thresh:end) = 0;
    img(1:start_point(2)-thresh,:) = 0;
    img(start_point(2)+thresh:end,:) = 0;
    [val,ind] = max(img(:));
    [p1,p2]=ind2sub(size(img),ind);
    x1(i)=p2;
    y1(i)=p1;
    start_point=[p2, p1];
end

% video 2_2
[a2, b2, c2, d2] = size(vidFrames2_2);                          
start_point = [350,247];                
x2=zeros(1,d2);
y2=zeros(1,d2);
for i=1:1:d2
    img = rgb2gray(vidFrames2_2(:,:,:,i));
    img(:,1:start_point(1)-thresh) = 0;
    img(:,start_point(1)+thresh:end) = 0;
    img(1:start_point(2)-thresh,:) = 0;
    img(start_point(2)+thresh:end,:) = 0;
    [val,ind] = max(img(:));
    [p1,p2]=ind2sub(size(img),ind);
    x2(i)=p2;
    y2(i)=p1;
    start_point=[p2, p1];
end

% video 3_2
[a3, b3, c3, d3] = size(vidFrames3_2);                           
start_point = [323,273];                
x3=zeros(1,d3);
y3=zeros(1,d3);
for i=1:1:d3
    img = rgb2gray(vidFrames3_2(:,:,:,i));
    img(:,1:start_point(1)-thresh) = 0;
    img(:,start_point(1)+thresh:end) = 0;
    img(1:start_point(2)-thresh,:) = 0;
    img(start_point(2)+thresh:end,:) = 0;
    [val,ind] = max(img(:));
    [p1,p2]=ind2sub(size(img),ind);
    x3(i)=p1;
    y3(i)=p2;
    start_point=[p2, p1];
end

% trim to same size
n = min(min(d1,d2),d3);
% x1=x1(1:n);
% y1=y1(1:n);
% x2=x2(1:n);
% y2=y2(1:n);
% x3=x3(1:n);
% y3=y3(1:n);

x1=x1(13:n);
y1=y1(13:n);
x2=x2(43:n+30);
y2=y2(43:n+30);
x3=x3(15:n+2);
y3=y3(15:n+2);

X=[x1;y1;x2;y2;x3;y3];

% PCA
[m,n]=size(X);
mn=mean(X,2);
X=X-repmat(mn,1,n);
Cx=(1/(n-1))*X*X';
[V,D]=eig(Cx);
lambda=diag(D);
[dummy, m_arrange]=sort(-1*lambda);
V=V(:,m_arrange);
Y=V'*X;

figure(1)
subplot(3,4,1)
plot(x1)
xlabel('Frame number')
ylabel('Position in X')
title('Cam 1')
subplot(3,4,2)
plot(y1)
xlabel('Frame number')
ylabel('Position in Y')
title('Cam 1')
subplot(3,4,5)
plot(x2)
xlabel('Frame number')
ylabel('Position in X')
title('Cam 2')
subplot(3,4,6)
plot(y2)
xlabel('Frame number')
ylabel('Position in Y')
title('Cam 2')
subplot(3,4,9)
plot(x3)
xlabel('Frame number')
ylabel('Position in X')
title('Cam 3')
subplot(3,4,10)
plot(y3)
xlabel('Frame number')
ylabel('Position in Y')
title('Cam 3')
subplot(3,4,[3,4])
plot(Y(1,:))
xlabel('Frame number')
ylabel('Position of the mass')
title('First principle component')
subplot(3,4,[7,8])
plot(Y(2,:))
xlabel('Frame number')
ylabel('Position of the mass')
title('Second principle component')
subplot(3,4,[11,12])
plot(Y(3,:))
xlabel('Frame number')
ylabel('Position of the mass')
title('Third principle component')
