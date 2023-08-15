% Created by Abhinaba Bhattacharjee, Purdue University, 6/29/2023

% This code is used to organise the Clinical Dataset of force based hand 
% manipulation motion sequences used in Manual Therapy as performed by 5 
% experienced Clinicians. The Raw Data contains Triaxial Forces, IMU data 
% (Accelerometer, Gyroscope and/or Magnetometer Data), which are sorted, 
% compiled and synthesized to generate Training Vectors and Testing Vectors 
% for deep learning model (LSTM-RNN) training and multimodal sequence 
% prediction.
%
%The Dataset is labelled with atleast 3 categories, which are "Strumming" and
%"JStroke" and "Random N/A" to perform linear and curved stroke based massage
%routines classification and classify the uncategorised sensor readings
%into "Random N/A" class. However the fundamental motion sequences
%"Strumming" and "JStroke" are further classified into "JStroke Correct" or
%"JStroke Incorrect" based on the performance of the therapist.

clc;
clear all;
%% Select the Path
% Select Foldername'ClinicalData' in the currect working directory
path = uigetdir('/Users/user/');
%% Select and enlist all the .csv files from subfolders in the working directory 

files = dir(fullfile(path,'**','*.csv'));
totalsize = size(files);
dataFile = cell(totalsize(1),4);
dataFileLbl = cell(totalsize(1),4);
classesLbl = cell(totalsize(1),4);
arrK = cell(totalsize(1),2);
arrTime = cell(totalsize(1),2);

for i = 1:size(files)
    csvFile = fullfile(files(i).folder,'\',files(i).name);
    Data = readtable(csvFile);
    datSz = size(Data);
    rst  = table2array(Data(:,13));
    time = table2array(Data(:,1));
    k = find(rst==1);
   
    rstIdx = k.';
    timeStamps = time([rstIdx]);
    arrK{i,1}(:,:) = files(i).name;
    arrK{i,2}(:,:) = rstIdx;
    arrTime{i,1}(:,:) = files(i).name;
    arrTime{i,2}(:,:) = timeStamps;


    dataTL = Data(1:k(1),:);
    lblDatTL = categorical(append(string(table2array(dataTL(:,21))),' ',string(table2array(dataTL(:,22)))));
    classesTL = categories(lblDatTL);

    dataUL= Data(k(2):k(3),:);
    lblDatUL = categorical(append(string(table2array(dataUL(:,21))),' ',string(table2array(dataUL(:,22)))));
    classesUL = categories(lblDatUL);

    dataLL= Data(k(4):datSz(1),:); 
    lblDatLL = categorical(append(string(table2array(dataLL(:,21))),' ',string(table2array(dataLL(:,22)))));
    classesLL = categories(lblDatLL);

    dataFile{i,1}(:,:) = files(i).name;
    dataFile{i,2}(:,:) = dataTL;
    dataFile{i,3}(:,:) = dataUL;
    dataFile{i,4}(:,:) = dataLL;

    dataFileLbl{i,1}(:,:) = files(i).name;
    dataFileLbl{i,2}(:,:) = lblDatTL;
    dataFileLbl{i,3}(:,:) = lblDatUL;
    dataFileLbl{i,4}(:,:) = lblDatLL;

    classesLbl{i,1}(:,:) = files(i).name;
    classesLbl{i,2}(:,:) = classesTL;
    classesLbl{i,3}(:,:) = classesUL;
    classesLbl{i,4}(:,:) = classesLL;
    
end

%% Prep Training datasets
% 11 colums of data Fx, Fy, Fz, F(Rez), Yaw, Pitch, Roll, AccX, AccY, AccZ,
% GyroRez
xTrain = cell(5,1);
yTrain = cell(5,1);

xValidate = cell(2,1);
yValidate = cell (2,1)
datIdx = [2:11,14]; % columns 2 to 11 and 14 (Force X Y Z Rez Yaw Pitch Roll Ax Ay Az GyroRMS)


xTrain{1}(:,:) = [table2array(dataFile{7,2}(:,datIdx));table2array(dataFile{7,3}(:,datIdx));...
            table2array(dataFile{5,2}(:,datIdx));table2array(dataFile{5,3}(:,datIdx));...
            table2array(dataFile{15,2}(:,datIdx));table2array(dataFile{15,3}(:,datIdx));...
            table2array(dataFile{13,2}(:,datIdx));table2array(dataFile{13,4}(:,datIdx))].';

xTrain{2}(:,:) = [table2array(dataFile{11,2}(:,datIdx));table2array(dataFile{11,3}(:,datIdx));...
            table2array(dataFile{9,2}(:,datIdx));table2array(dataFile{9,3}(:,datIdx));...
            table2array(dataFile{19,2}(:,datIdx));table2array(dataFile{19,3}(:,datIdx));...
            table2array(dataFile{17,2}(:,datIdx));table2array(dataFile{17,3}(:,datIdx))].';

xTrain{3}(:,:) = [table2array(dataFile{3,2}(:,datIdx));table2array(dataFile{3,3}(:,datIdx));...
            table2array(dataFile{1,2}(:,datIdx));table2array(dataFile{1,3}(:,datIdx));...
            table2array(dataFile{16,2}(:,datIdx));table2array(dataFile{14,2}(:,datIdx))].';

xTrain{4}(:,:) = [table2array(dataFile{8,2}(:,datIdx));table2array(dataFile{8,3}(:,datIdx));...
            table2array(dataFile{10,2}(:,datIdx));table2array(dataFile{10,3}(:,datIdx));...
            table2array(dataFile{16,3}(:,datIdx));table2array(dataFile{14,3}(:,datIdx));...
            table2array(dataFile{2,3}(:,datIdx));table2array(dataFile{4,3}(:,datIdx))].';

xTrain{5}(:,:) = [table2array(dataFile{6,2}(:,datIdx));table2array(dataFile{6,3}(:,datIdx));...
            table2array(dataFile{12,2}(:,datIdx));table2array(dataFile{12,3}(:,datIdx));...
            table2array(dataFile{18,2}(:,datIdx));table2array(dataFile{18,3}(:,datIdx));...
            table2array(dataFile{19,4}(:,datIdx));table2array(dataFile{16,4}(:,datIdx))].';



yTrain{1}(:,:) = [dataFileLbl{7,2};dataFileLbl{7,3};dataFileLbl{5,2};dataFileLbl{5,3};...
            dataFileLbl{15,2};dataFileLbl{15,3};dataFileLbl{13,2};dataFileLbl{13,4}].';

yTrain{2}(:,:) = [dataFileLbl{11,2};dataFileLbl{11,3};dataFileLbl{9,2};dataFileLbl{9,3};...
            dataFileLbl{19,2};dataFileLbl{19,3};dataFileLbl{17,2};dataFileLbl{17,3}].';

yTrain{3}(:,:) = [dataFileLbl{3,2};dataFileLbl{3,3};dataFileLbl{1,2};...
            dataFileLbl{1,3};dataFileLbl{16,2};dataFileLbl{14,2}].';

yTrain{4}(:,:) = [dataFileLbl{8,2};dataFileLbl{8,3};dataFileLbl{10,2};dataFileLbl{10,3};...
            dataFileLbl{16,3};dataFileLbl{14,3};dataFileLbl{2,3};dataFileLbl{4,3}].';

yTrain{5}(:,:) = [dataFileLbl{6,2};dataFileLbl{6,3};dataFileLbl{12,2};dataFileLbl{12,3};...
            dataFileLbl{18,2};dataFileLbl{18,3};dataFileLbl{19,4};dataFileLbl{16,4}].';

%% Prepare Testing Datasets

xTest = cell(1,1);
yTest = cell(1,1);

xTest{1}(:,:) = [table2array(dataFile{1,4}(:,datIdx));table2array(dataFile{2,4}(:,datIdx));...
            table2array(dataFile{3,4}(:,datIdx));table2array(dataFile{4,4}(:,datIdx));...
            table2array(dataFile{5,4}(:,datIdx));table2array(dataFile{6,4}(:,datIdx));...
            table2array(dataFile{7,4}(:,datIdx));table2array(dataFile{8,4}(:,datIdx));...
            table2array(dataFile{9,4}(:,datIdx));table2array(dataFile{10,4}(:,datIdx));...
            table2array(dataFile{11,4}(:,datIdx));table2array(dataFile{12,4}(:,datIdx));...
            table2array(dataFile{13,3}(:,datIdx));table2array(dataFile{14,4}(:,datIdx));...
            table2array(dataFile{15,4}(:,datIdx));table2array(dataFile{4,2}(:,datIdx));...
            table2array(dataFile{17,4}(:,datIdx));table2array(dataFile{18,4}(:,datIdx));...
            table2array(dataFile{2,2}(:,datIdx))].';

yTest{1}(:,:) = [dataFileLbl{1,4};dataFileLbl{2,4};dataFileLbl{3,4};dataFileLbl{4,4};...
            dataFileLbl{5,4};dataFileLbl{6,4};dataFileLbl{7,4};dataFileLbl{8,4};...
            dataFileLbl{9,4};dataFileLbl{10,4};dataFileLbl{11,4};dataFileLbl{12,4};...
            dataFileLbl{13,3};dataFileLbl{14,4};dataFileLbl{15,4};dataFileLbl{4,2};...
            dataFileLbl{17,4};dataFileLbl{18,4};dataFileLbl{2,2}].';

%% Define the features of the training and testing Datasets

% Eleven features of data 1-Fx, 2-Fy, 3-Fz, 4-F(Rez), 5-Yaw, 6-Pitch,
% 7-Roll, 8-AccX, 9-AccY, 10-AccZ, 11-GyroRez. 
% Use only one of the feature sets for a single training and testing
% session


% % % % Features = [1:3];% Force Features
% % % % Features = [5:7];% Yaw, Pitch, Roll Features
% % % % Features = [8:10];% Acceleration Features
% % % % Features = [1:3,5:7];% Force & YPR Features
% % % % Features = [1:4,11];% Force & ForceRMS & GyroRMS Features
% % % % Features = [1:3,8:10];% Force & Acceleration Features
% % % % Features = [1:3,8:11];% Force & Acceleration Features & Gyro
% % % % Features = [4,11];% Resultant Force & Gyro Resultant
% % % % Features = [1:11];% All Features

xTrainFeatures = cell(5,1);

for j = 1:length(xTrain)
    xTrainFeatures{j} = xTrain{j}(Features,:);

end

% xValFeatures = cell(2,1);
% 
% for s = 1:length(xValidate)
%     xValFeatures{s} = xValidate{s}(Features,:);
% 
% end
%%
xTestFeatures = cell(1,1);% intialize testing feature vector
xTestFeatures{1} = xTest{1}(Features,:);% assigning testing vector 

%% Proportion Statistics
figure
for Q = 1:numel(xTrain)

    X = xTrain{Q}(11,:);
    classes = categories(yTrain{1});
    strtIdx=1;
    idArr = zeros(1,5);
    
    
    for p = 1:numel(classes)
        label = classes(p);
        id = find(yTrain{Q} == label);
        idArr(1,p) = length(id);
        if p>1
            strtIdx = endIdx+1;
            endIdx = strtIdx+length(id)-1;
        else
            endIdx = length(id);
        end
        hold on
        subplot(6,1,Q)
        plot([strtIdx:endIdx],X(id))
    end
    hold off
    
    xlabel("Time Step (10 ms)")
    ylabel("Degrees/Sec")
    str = sprintf("Proportion Statistics of Training Vector %d, Feature GyroRMS", Q);
    title(str)
    
end

X = xTest{1}(11,:);
    classes = categories(yTrain{1});
    strtIdx=1;
    idArr = zeros(1,5);
    
    
    for p = 1:numel(classes)
        label = classes(p);
        id = find(yTest{1} == label);
        idArr(1,p) = length(id);
        if p>1
            strtIdx = endIdx+1;
            endIdx = strtIdx+length(id)-1;
        else
            endIdx = length(id);
        end
        hold on
        subplot(6,1,6)
        plot([strtIdx:endIdx],X(id))
    end
    hold off
    
    xlabel("Time Step (10 ms)")
    ylabel("Degrees/Sec")
    str = sprintf("Proportion Statistics of Testing Vector, Feature GyroRMS");
    title(str)


legend(classes,'Location','northwest')


%% Define the LSTM-RNN Network

numFeatures = length(Features);
numHiddenUnits = 200;
numClasses = 5;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% %%
% options = trainingOptions('adam', ...
%     'MaxEpochs',200, ...
%     'GradientThreshold',2, ...
%     'Verbose',0, ...
%     'Plots','training-progress');

%%
options = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    ExecutionEnvironment="gpu", ...
    SequenceLength="longest", ...
    GradientThreshold=1, ...
    Shuffle="every-epoch", ...    
    Plots="training-progress", ...
    Verbose=0);
%% Train the network
net = trainNetwork(xTrainFeatures,yTrain,layers,options);

%% Classify the test data and get Classification Accuracy metrics
[YPred,scores] = classify(net,xTestFeatures{1});
%%% Accuracy 
acc = sum(YPred == yTest{1})./numel(yTest{1})
plotconfusion(yTest{1},YPred)

figure
hold on
%accuracy = mean(YPred == yTest{1});
rocObj = rocmetrics(yTest{1},scores.',categorical(classes));
[FPR1,TPR1,Thresholds1,AUC1] = average(rocObj,"macro");
[FPR2,TPR2,Thresholds2,AUC2] = average(rocObj,"micro");

subplot(1,2,1)
plot(rocObj,AverageROCType="macro", ClassNames=[])
subplot(1,2,2)
plot(rocObj,AverageROCType="micro", ClassNames=[])
hold off
%legend(ClassNames,'Location','northwest')
str = sprintf("Accuracy Metrics macro AUC Macro=%0.2f, AUC Micro = %0.2f, Mean True Positive Rate  = %0.4f ", AUC1, AUC2, mean(TPR2));
disp(str);


          

         



