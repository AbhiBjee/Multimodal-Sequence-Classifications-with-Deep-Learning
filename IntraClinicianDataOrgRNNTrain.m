% Created by Abhinaba Bhattacharjee, Purdue University, 6/29/2023

% This code is used to train and test force based hand manipulation motion sequences used in Manual Therapy as performed by an individual experienced clinicians. 
% The Data acquired from the manipulating the left side of the body is selected for training while that of the right side is selected for testing. 
% The Raw Data contains Triaxial Forces, IMU data (Accelerometer, Gyroscope and/or Magnetometer Data), which are segregated into Training Vector and Testing Vector 
% for deep learning model (LSTM-RNN) training and multimodal sequence prediction.
%
%The Dataset is labelled with atleast 2 categories, which are either ("Strumming" and "Random N/A") to perform linear stroke based massage 
% or ("JStroke" and "Random N/A") curved motion based massage on different anatomical regions of the body especially - Thoracolumbar, Upper Thigh and Calf regions

%% workspace clear
clc;
clear all;
%% load data files
% Select either of C_11 or C_1 in the Foldername 'ClinicalData' in the currect working directory. 

path = uigetdir('/Users/user/');
files = dir(fullfile(path,'**','*.csv'));
totalsize = length(files);
dataFile = cell(totalsize(1),2);
dataLblFile = cell(totalsize(1),2);
idx = [1:11,14];

for i = 1:totalsize
    csvFile = fullfile(files(i).folder,'\',files(i).name);
    Data = readtable(csvFile);

    dataFile{i,1}(:,:) = files(i).name;
    dataFile{i,2}(:,:) = table2array(Data(:,idx));
    
    lblDat= categorical(append(string(table2array(Data(:,21))),' ',string(table2array(Data(:,22)))));
    dataLblFile{i,1}(:,:) = files(i).name;
    dataLblFile{i,2}(:,:) = lblDat;
end

%% Prepare training and testing data files
% Data files sequence column indices:
%Time, 2-FrcX, 3-FrcY, 4-FrcZ, 5-FrcRez, 6-Yaw, 7-Pitch, 8-Roll, 9-AccX, 
% 10-AccY,11-AccZ,12-GyroRez

Features = [2:4,9:11,12];% Define the 7 characterisitic features (Triaxial forces, Triaxial Acceleration and GyroRMS) 

%%%%% training data files
xTrain = cell(1,1);
yTrain = cell(1,1);

xTrain{1}(:,:) = [dataFile{1,2}(:,Features);dataFile{3,2}(:,Features)].';
yTrain{1}(:,:) = [dataLblFile{1,2};dataLblFile{3,2}].';


%%%%% testing data files
xTest = cell(1,1);
yTest = cell(1,1);

xTest{1}(:,:) = [dataFile{2,2}(:,Features);dataFile{4,2}(:,Features)].';
yTest{1}(:,:) = [dataLblFile{2,2};dataLblFile{4,2}].';

% xTest{1}(:,:) = [dataFile{4,2}(:,Features)].';
% yTest{1}(:,:) = [dataLblFile{4,2}].';

%% Define the LSTM-RNN Network

class = categories(yTrain{1});

numFeatures = length(Features);
numHiddenUnits = 200;
numClasses = length(class);

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%%
options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    ExecutionEnvironment="gpu", ...
    SequenceLength="longest", ...
    GradientThreshold=2, ...
    Shuffle="every-epoch", ...    
    Plots="training-progress", ...
    Verbose=0);
%%
net = trainNetwork(xTrain,yTrain,layers,options);

%% Classify the test data and view the sensitivity and ROC metrics

[YPred,scores] = classify(net,xTest{1});
%%% Accuracy 
acc = sum(YPred == yTest{1})./numel(yTest{1})
plotconfusion(yTest{1},YPred)
classes = categories(YPred);

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