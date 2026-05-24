% Test of YOLO detector

clc
close all

load detectorYolo2.mat

%% Detection threshold
threshold = 0.35;

%% Read test data
dataArray = readall(testData);
dataTable = cell2table(dataArray);

dataImages = dataTable{:,1};

%% Prepare results table
results = table('Size',[height(dataImages) 3],...
'VariableTypes',{'cell','cell','cell'},...
'VariableNames',{'Boxes','Scores','Labels'});

%% Detect objects
for i = 1:height(dataImages)

I = dataImages{i};

[bboxes,scores,labels] = detect(detectorYolo2,I,'Threshold',threshold);

if ~isempty(bboxes)

[selectedBboxes,selectedScores,idx] = selectStrongestBbox(bboxes,scores,...
'OverlapThreshold',0.3);

bboxes = selectedBboxes;
scores = selectedScores;
labels = labels(idx);

I = insertObjectAnnotation(I,'Rectangle',bboxes,cellstr(labels));

figure(i)
imshow(I)

pause(0.3)

end

results.Boxes{i} = floor(bboxes);
results.Scores{i} = scores;
results.Labels{i} = labels;

end

%% Evaluation

[ap,recall,precision] = evaluateDetectionPrecision(results,testData,threshold);
[am,fppi,missRate] = evaluateDetectionMissRate(results,testData,threshold);

%% Fix cell arrays

if iscell(recall)
recall = recall{1};
end

if iscell(precision)
precision = precision{1};
end

if iscell(fppi)
fppi = fppi{1};
end

if iscell(missRate)
missRate = missRate{1};
end

%% Plot metrics

figure
set(gcf,'Position',[200 200 600 500])

subplot(1,2,1)

plot(recall,precision,'b-','LineWidth',2)

xlabel('Recall')
ylabel('Precision')
title(sprintf('Average Precision = %.2f',ap))
grid on

subplot(1,2,2)

loglog(fppi,missRate,'b-','LineWidth',2)

xlabel('False Positives Per Image')
ylabel('Log Average Miss Rate')
title(sprintf('Log Average Miss Rate = %.2f',am))
grid on