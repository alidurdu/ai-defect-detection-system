% This file builds and trains a YOLOv2 object detector

%% Input size
inputSize = [224 224 3];

%% Image input layer
inputLayer = imageInputLayer(inputSize,'Name','input');

%% Filter size
filterSize = [3 3];

%% Feature extraction layers
featureLayers = [
    convolution2dLayer(filterSize,16,'Padding',1,'Name','conv_1')
    batchNormalizationLayer('Name','BN1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool1')

    convolution2dLayer(filterSize,32,'Padding',1,'Name','conv_2')
    batchNormalizationLayer('Name','BN2')
    reluLayer('Name','relu_2')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool2')

    convolution2dLayer(filterSize,64,'Padding',1,'Name','conv_3')
    batchNormalizationLayer('Name','BN3')
    reluLayer('Name','relu_3')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool3')

    convolution2dLayer(filterSize,128,'Padding',1,'Name','conv_4')
    batchNormalizationLayer('Name','BN4')
    reluLayer('Name','relu_4')
];

%% Combine layers
layers = [
    inputLayer
    featureLayers
];

lgraph = layerGraph(layers);

%% Classes
classNames = {'Oil','Scratch','Stain'};

%% Anchor boxes
numAnchors = 10;
[anchorBoxes,meanIoU] = estimateAnchorBoxes(scaledTrainingData,numAnchors);

%% Create YOLOv2 layers
lgraph = yolov2Layers(inputSize,numel(classNames),anchorBoxes,lgraph,'relu_4');

%% Training options
options = trainingOptions('sgdm',...
    'MaxEpochs',20,...
    'MiniBatchSize',16,...
    'InitialLearnRate',1e-3,...
    'Shuffle','every-epoch',...
    'VerboseFrequency',1);

%% Random seed
rng(0)

%% Train detector
[detectorYolo2,info] = trainYOLOv2ObjectDetector(scaledTrainingData,lgraph,options);

%% Save model
save detectorYolo2.mat detectorYolo2

disp("Training completed successfully")