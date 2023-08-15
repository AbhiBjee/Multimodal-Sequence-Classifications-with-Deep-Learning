%% workspace clear
clc;
clear all;
%% load data files
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

%Time, 2-FrcX, 3-FrcY, 4-FrcZ, 5-FrcRez, 6-Yaw, 7-Pitch, 8-Roll, 9-AccX, 10-AccY,11-AccZ,
%12-GyroRez

Features = [2:4,9:11,12];

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

%% Classify the test data
% clear xTest, yTest;
% xTest{1}(:,:) = [dataFile{4,2}(:,Features)].';
% yTest{1}(:,:) = [dataLblFile{4,2}].';

YPred = classify(net,xTest{1});
%%% Accuracy 
acc = sum(YPred == yTest{1})./numel(yTest{1})
plotconfusion(yTest{1},YPred)