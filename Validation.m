% This file validates the network
clc
close all

% Validation data
% Read all the data from "validationData" into an array "dataArray"
dataArray = readall(validationData);

% Transform the array "dataArray" into a table "dataTable"
dataTable = cell2table(dataArray);

% Extract images from table
dataImages = dataTable{:,1};

% Detect objects on validation datastore
detectionResults = detect(detectorYolo2,validationData,'Threshold',0.2);

% Evaluate detection precision
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults, validationData);

% Create table for results
results = table('Size',[height(dataImages) 3],...
    'VariableTypes',{'cell','cell','cell'},...
    'VariableNames',{'Boxes','Scores','Labels'});

% Loop through validation images
for i = 1:height(dataImages)

    I = dataImages{i};

    % Run detector
    [bboxes,scores,labels] = detect(detectorYolo2,I);

    if ~isempty(bboxes)
        I = insertObjectAnnotation(I,'Rectangle',bboxes,cellstr(labels));
        figure(i)
        imshow(I)
        pause(0.5)
    end

    % Save results
    results.Boxes{i} = floor(bboxes);
    results.Scores{i} = scores;
    results.Labels{i} = labels;

end

% Evaluation metrics
threshold = 0.2;

[ap, recall, precision] = evaluateDetectionPrecision(results, validationData,threshold);
[am,fppi,missRate] = evaluateDetectionMissRate(results, validationData,threshold);

% Convert cell arrays if necessary
if iscell(recall)
    recall = recall{1};
end

if iscell(precision)
    precision = precision{1};
end

if iscell(missRate)
    missRate = missRate{1};
end

if iscell(fppi)
    fppi = fppi{1};
end

% Plot evaluation metrics
figure
set(gcf,'Position',[200 200 500 500])

subplot(1,2,1)
plot(recall,precision,'b-','LineWidth',2)
xlabel('Recall')
ylabel('Precision')
title(sprintf('Average Precision = %.2f', ap))
grid on

subplot(1,2,2)
loglog(fppi, missRate,'b-','LineWidth',2)
xlabel('False Positives Per Image')
ylabel('Log Average Miss Rate')
title(sprintf('Log Average Miss Rate = %.2f', am))
grid on