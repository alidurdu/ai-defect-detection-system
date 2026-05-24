% This file loads and prepares the data for training and validating the network
clear all
%% Load gTruth (300x4 table)
load gTruth.mat

%% Shuffle rows
numImages = height(gTruth);
shuffledIndices = randperm(numImages);

%% Split ratios
trainRatio = 0.6;
valRatio   = 0.1;

%% Compute sizes
idxTrain = floor(trainRatio * numImages);
idxVal   = floor(valRatio   * numImages);

%% Training set
trainingIdx     = shuffledIndices(1:idxTrain);
trainingDataTbl = gTruth(trainingIdx,:);

%% Validation set
validationIdx     = shuffledIndices(idxTrain+1 : idxTrain+idxVal);
validationDataTbl = gTruth(validationIdx,:);

%% Test set
testIdx     = shuffledIndices(idxTrain+idxVal+1 : end);
testDataTbl = gTruth(testIdx,:);

%% Create Image Datastores
imdsTrain = imageDatastore(trainingDataTbl.imageFilename);
imdsValidation = imageDatastore(validationDataTbl.imageFilename);
imdsTest = imageDatastore(testDataTbl.imageFilename);

%% Create Box Label Datastores
bldsTrain = boxLabelDatastore(trainingDataTbl(:,2:end));
bldsValidation = boxLabelDatastore(validationDataTbl(:,2:end));
bldsTest = boxLabelDatastore(testDataTbl(:,2:end));

%% Combine Image + Labels
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

%% Resize to 224x224
scaledTrainingData = transform(trainingData,@scaleGT);
scaledValidationData = transform(validationData,@scaleGT);
scaledTestData = transform(testData,@scaleGT);

%% Display dataset sizes
disp("Training size:")
disp(height(trainingDataTbl))

disp("Validation size:")
disp(height(validationDataTbl))

disp("Test size:")
disp(height(testDataTbl))

%% Scale function
function data = scaleGT(data)
    targetSize = [224 224];
    
    % Resize image
    scale = targetSize ./ size(data{1},[1 2]);
    data{1} = imresize(data{1},targetSize);
    
    % Resize bounding boxes
    data{2} = bboxresize(data{2},scale);
end