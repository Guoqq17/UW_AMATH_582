close all; clear; clc
%% import data
[images_train, labels_train] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[images_test, labels_test] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');

%% %% part I
[a_train,b_train,c_train]=size(images_train);
[a_test,b_test,c_test]=size(images_test);
data_train = zeros(a_train*b_train, c_train);
data_test = zeros(a_test*b_test, c_test);
for i=1:1:c_train
    data_train(:,i)=reshape(images_train(:,:,i),a_train*b_train, 1);
end
for i=1:1:c_test
    data_test(:,i)=reshape(images_test(:,:,i),a_test*b_test, 1);
end
%% SVD
[U,S,V]=svd(double(data_train),'econ');
proj=(S*V')';
%% singular value spectrum
plot(diag(S)/sum(diag(S)), '-o')
xlabel('Singular value','Fontsize',12)
ylabel('Proportion','Fontsize',12)
%% reconstruct under different v-modes
figure(2)
num_recon = [5, 10, 30, 50, 70, 100, 200, 400, 600, 782];
for i=1:1:length(num_recon)
    img_recon = U*S(:,1:num_recon(i))*V(:,1:num_recon(i)).';
    img=uint8(reshape(img_recon(:,1), a_train, b_train));
    subplot(2,5,i)
    imshow(img)
    title(['No. of SV:', num2str(num_recon(i))])
end
%% V-modes visulization
figure(3);
colormap jet
for i=0:1:9
    ind = find(labels_train==i);
    scatter3(V(ind,2),V(ind,3),V(ind,5),20,labels_train(ind),'.')
    hold on
end
xlabel('Column 2 of V')
ylabel('Column 3 of V')
zlabel('Column 5 of V')
legend({'0','1','2','3','4','5','6','7','8','9'});

%% %% Part II
%% LDA, 2 digits example, 0 and 4, find the best number of features
num_v=[2,3,4,5,10,30,50];
accuracy_v=zeros(1,length(num_v));
for i=1:1:length(num_v)
    x1_train=proj(labels_train==0,2:num_v(i));
    x2_train=proj(labels_train==4,2:num_v(i));
    [len1,temp]=size(x1_train);
    [len2,temp]=size(x2_train);
    xtrain=[x1_train; x2_train];
    ctrain=[0*ones(len1,1); 4*ones(len2,1)];

    xtest_temp=(U'*data_test)';
    x1_test=xtest_temp(labels_test==0,2:num_v(i));
    x2_test=xtest_temp(labels_test==4,2:num_v(i));
    [len1,temp]=size(x1_test);
    [len2,temp]=size(x2_test);
    xtest=[x1_test; x2_test];
    ctest=[0*ones(len1,1); 4*ones(len2,1)];

    pre=classify(xtest,xtrain,ctrain);

    errorNum=sum(abs(ctest-pre)>0);
    accuracy_v(i)=1-errorNum/length(ctest);
end
figure(4)
plot(num_v-1, accuracy_v, '-o')
xlabel('Number of V-modes','Fontsize',12)
ylabel('Accuracy on test data','Fontsize',12)

%% LDA, compare different digit pairs
accuracy_lda=zeros(10,10);
for i=0:1:8
    for j=i+1:1:9
        x1_train=proj(labels_train==i,2:10);
        x2_train=proj(labels_train==j,2:10);
        [len1,temp]=size(x1_train);
        [len2,temp]=size(x2_train);
        xtrain=[x1_train; x2_train];
        ctrain=[i*ones(len1,1); j*ones(len2,1)];

        xtest_temp=(U'*data_test)';
        x1_test=xtest_temp(labels_test==i,2:10);
        x2_test=xtest_temp(labels_test==j,2:10);
        [len1,temp]=size(x1_test);
        [len2,temp]=size(x2_test);
        xtest=[x1_test; x2_test];
        ctest=[i*ones(len1,1); j*ones(len2,1)];

        pre=classify(xtest,xtrain,ctrain);

        errorNum=sum(abs(ctest-pre)>0);
        accuracy_lda(i+1,j+1)=1-errorNum/length(ctest);
    end
end

%% LDA, 3 digits example, 0, 4 and 7
x1_train=proj(labels_train==0,2:10);
x2_train=proj(labels_train==4,2:10);
x3_train=proj(labels_train==7,2:10);
[len1,temp]=size(x1_train);
[len2,temp]=size(x2_train);
[len3,temp]=size(x3_train);
xtrain=[x1_train; x2_train; x3_train];
ctrain=[0*ones(len1,1); 4*ones(len2,1); 7*ones(len3,1)];

xtest_temp=(U'*data_test)';
x1_test=xtest_temp(labels_test==0,2:10);
x2_test=xtest_temp(labels_test==4,2:10);
x3_test=xtest_temp(labels_test==7,2:10);
[len1,temp]=size(x1_test);
[len2,temp]=size(x2_test);
[len3,temp]=size(x3_test);
xtest=[x1_test; x2_test; x3_test];
ctest=[0*ones(len1,1); 4*ones(len2,1); 7*ones(len3,1)];

pre=classify(xtest,xtrain,ctrain);

errorNum=sum(abs(ctest-pre)>0);
accuracy_047=1-errorNum/length(ctest)

figure(5)
bar(pre)
xlabel('case number (first 980 are 0, the following 982 are 4, the rest are 7)')
ylabel('prediction results')

%% LDA, all digits
xtrain=proj(:,2:10);
xtest_temp=(U'*data_test)';
xtest=xtest_temp(:,2:10);

pre=classify(xtest,xtrain,labels_train);

errorNum=sum(abs(labels_test-pre)>0);
accuracy_lda=1-errorNum/length(labels_test)

%% CT, all digits
xtrain=proj(:,2:10);
xtest_temp=(U'*data_test)';
xtest=xtest_temp(:,2:10);
Mdl = fitctree(xtrain,labels_train,'OptimizeHyperparameters','auto');

pre=predict(Mdl,xtest);

errorNum=sum(abs(labels_test-pre)>0);
accuracy_ct=1-errorNum/length(labels_test)

%% SVM, all digits
xtrain=proj(:,2:10)/max(max(S));
xtest=xtest_temp(:,2:10)/max(max(S));

SVMModels = cell(10,1);
classes = 0:1:9;
rng(1); % For reproducibility
for j = 1:numel(classes)
    indx = labels_train==classes(j); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(xtrain,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
end
for j = 1:numel(classes)
    [~,score] = predict(SVMModels{j},xtest);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,maxScore] = max(Scores,[],2);
errorNum=sum(abs(labels_test+1-maxScore)>0);
accuracy_ct=1-errorNum/length(labels_test)

%% easiest and hardest pairs
pair = [0,1]; % easiest
% pair = [4,9]; % hardest

x1_train=proj(labels_train==pair(1),2:10);
x2_train=proj(labels_train==pair(2),2:10);
[len1,temp]=size(x1_train);
[len2,temp]=size(x2_train);
xtrain=[x1_train; x2_train];
ctrain=[i*ones(len1,1); j*ones(len2,1)];

xtest_temp=(U'*data_test)';
x1_test=xtest_temp(labels_test==pair(1),2:10);
x2_test=xtest_temp(labels_test==pair(2),2:10);
[len1,temp]=size(x1_test);
[len2,temp]=size(x2_test);
xtest=[x1_test; x2_test];
ctest=[i*ones(len1,1); j*ones(len2,1)];

% CT
xtrain=proj(:,2:10);
xtest_temp=(U'*data_test)';
xtest=xtest_temp(:,2:10);
Mdl_ct = fitctree(xtrain,labels_train,'OptimizeHyperparameters','auto');

pre=predict(Mdl_ct,xtest);

errorNum=sum(abs(ctest-pre)>0);
accuracy_ct_2=1-errorNum/length(ctest)

% SVM
rng default
Mdl_svm = fitcsvm(xtrain,ctrain,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'))

pre=predict(Mdl_svm,xtest);

errorNum=sum(abs(ctest-pre)>0);
accuracy_svm_2=1-errorNum/length(ctest);
