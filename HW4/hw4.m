close all; clear; clc
[images, labels] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
%% %% part I
[a,b,c]=size(images);
Nums = zeros(a*b, c);

for i=1:1:c
    Nums(:,i)=reshape(images(:,:,i),a*b, 1);
end
%% SVD
[U,S,V]=svd(double(Nums),'econ');

%% singular value spectrum
plot(diag(S)/sum(diag(S)), '-o')
xlabel('Singular value','Fontsize',12)
ylabel('Proportion','Fontsize',12)

%% reconstruct
figure(2)
num_recon = [5, 10, 30, 50, 70, 100, 200, 400, 600, 782];
for i=1:1:length(num_recon)
    img_recon = U*S(:,1:num_recon(i))*V(:,1:num_recon(i)).';
    img=uint8(reshape(img_recon(:,4), a, b));
    subplot(2,5,i)
    imshow(img)
    title(['No. of SV:', num2str(num_recon(i))])
end

%% V-modes
figure(3);
colormap jet
for i=0:1:9
    ind = find(labels==i);
    scatter3(V(ind,2),V(ind,3),V(ind,5),20,labels(ind),'.')
    hold on
end
xlabel('Column 2 of V')
ylabel('Column 3 of V')
zlabel('Column 5 of V')
legend({'0','1','2','3','4','5','6','7','8','9'});

%% %% Part II
%% 2 digits example, 0 and 4, find the best number of V-modes
num_v=[2,3,4,5,10,30,50];
accuracy_v=zeros(1,length(num_v));
for i=1:1:length(num_v)
    episode=20;
    accu_temp=zeros(1,episode);
    for j=1:1:episode
        x1=V(labels==0,2:num_v(i));
        x2=V(labels==4,2:num_v(i));
        q1=randperm(length(x1));
        q2=randperm(length(x2));
        len1=round(length(x1)*0.8);
        len2=round(length(x2)*0.8);

        xtrain=[x1(q1(1:len1),:); x2(q2(1:len2),:)];
        xtest=[x1(q1(len1+1:end),:); x2(q2(len2+1:end),:)];

        ctrain=[0*ones(len1,1); 4*ones(len2,1)];
        ctest=[0*ones(length(x1)-len1,1); 4*ones(length(x2)-len2,1)];

        pre=classify(xtest,xtrain,ctrain);
        
        errorNum=sum(abs(ctest-pre)>0);
        accu_temp(j)=1-errorNum/length(ctest);
    end
    accuracy_v(i)=mean(accu_temp);
end

figure(4)
plot(num_v-1, accuracy_v, '-o')
xlabel('Number of V-modes','Fontsize',12)
ylabel('Average accuracy (20 times)','Fontsize',12)

%% compare different digit pairs
accuracy_lda=zeros(10,10);
for i=0:1:8
    for j=i+1:1:9
        episode=20;
        accu_temp=zeros(1,episode);
        for k=1:1:episode
            x1=V(labels==i,2:5);
            x2=V(labels==j,2:5);
            q1=randperm(length(x1));
            q2=randperm(length(x2));
            len1=round(length(x1)*0.8);
            len2=round(length(x2)*0.8);

            xtrain=[x1(q1(1:len1),:); x2(q2(1:len2),:)];
            xtest=[x1(q1(len1+1:end),:); x2(q2(len2+1:end),:)];
            ctrain=[i*ones(len1,1); j*ones(len2,1)];
            ctest=[i*ones(length(x1)-len1,1); j*ones(length(x2)-len2,1)];
            pre=classify(xtest,xtrain,ctrain);

            errorNum=sum(abs(ctest-pre)>0);
            accu_temp(k)=1-errorNum/length(ctest);
        end
        accuracy_lda(i+1,j+1)=mean(accu_temp);
    end
end

%% 3 digits example, 0, 4 and 8
episode=20;
accu_temp=zeros(1,episode);
for j=1:1:episode
    x1=V(labels==0,2:5);
    x2=V(labels==4,2:5);
    x3=V(labels==7,2:5);
    q1=randperm(length(x1));
    q2=randperm(length(x2));
    q3=randperm(length(x3));
    len1=round(length(x1)*0.8);
    len2=round(length(x2)*0.8);
    len3=round(length(x3)*0.8);

    xtrain=[x1(q1(1:len1),:); x2(q2(1:len2),:); x3(q3(1:len3),:)];
    xtest=[x1(q1(len1+1:end),:); x2(q2(len2+1:end),:); x3(q3(len3+1:end),:)];

    ctrain=[0*ones(len1,1); 4*ones(len2,1); 7*ones(len3,1)];
    ctest=[0*ones(length(x1)-len1,1); 4*ones(length(x2)-len2,1); 7*ones(length(x3)-len3,1)];

    pre=classify(xtest,xtrain,ctrain);
    errorNum=sum(abs(ctest-pre)>0);
    accu_temp(j)=1-errorNum/length(ctest);
end
mean(accu_temp)

figure(5)
bar(pre)
xlabel('case number (first 1185 are 0, the following 1168 are 4, the rest are 7)')
ylabel('prediction results')

%% SVM classifier with training data, labels and test set
accuracy_svm=zeros(10,10);
for i=0:1:8
    for j=i+1:1:9
        episode=20;
        accu_temp=zeros(1,episode);
        for k=1:1:episode
            x1=[V(labels==i,2),V(labels==i,3),V(labels==i,5)];
            x2=[V(labels==j,2),V(labels==j,3),V(labels==j,5)];
            q1=randperm(length(x1));
            q2=randperm(length(x2));
            len1=round(length(x1)*0.8);
            len2=round(length(x2)*0.8);

            xtrain=[x1(q1(1:len1),:); x2(q2(1:len2),:)];
            xtest=[x1(q1(len1+1:end),:); x2(q2(len2+1:end),:)];
            ctrain=[i*ones(len1,1); j*ones(len2,1)];
            ctest=[i*ones(length(x1)-len1,1); j*ones(length(x2)-len2,1)];
            
            Mdl = fitcsvm(xtrain,ctrain,'KernelFunction','polynomial');
            pre = predict(Mdl,xtest);

            errorNum=sum(abs(ctest-pre)>0);
            accu_temp(k)=1-errorNum/length(ctest);
        end
        accuracy_svm(i+1,j+1)=mean(accu_temp);
    end
end

%% classification tree on fisheriris data
%% test maximum split numbers
MaxSplits=[10,20,30,40,50,100,200];
accuracy=zeros(1,length(MaxSplits));
X=[V(:,2),V(:,3),V(:,5)];
for i=1:1:length(MaxSplits)
    tree=fitctree(X,labels,'MaxNumSplits',MaxSplits(i),'CrossVal','on');
    classError = kfoldLoss(tree);
    accuracy(i)=1-classError;
end
%% test all pairs
accuracy_tree=zeros(10,10);
for i=0:1:8
    for j=i+1:1:9
        x1=[V(labels==i,2),V(labels==i,3),V(labels==i,5)];
        x2=[V(labels==j,2),V(labels==j,3),V(labels==j,5)];
        len1=length(x1);
        len2=length(x2);

        xtrain=[x1; x2];
        ctrain=[i*ones(len1,1); j*ones(len2,1)];

        tree=fitctree(xtrain,ctrain,'MaxNumSplits',40,'CrossVal','on');
        classError = kfoldLoss(tree);
        accuracy_tree(i+1,j+1)=1-classError;
    end
end



