clc
clear all
close all

%% PART 1: Baseline Classifier
%% Create image data store
imds = imageDatastore('fundus', ...
    'IncludeSubfolders',true,'LabelSource','foldernames')           %#ok

imds.ReadFcn = @ResizeImages;
% Count number of images per label and save the number of classes
labelCount = countEachLabel(imds);
numClasses = height(labelCount);

% Determine the smallest amount of images in a category
minSetCount = min(labelCount{:,2});

% Limit the number of images to reduce the time it takes
% run this example.
maxNumImages = 52;
minSetCount = min(maxNumImages,minSetCount);

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

[imdsTrainingSet, imdsValidationSet] = splitEachLabel(imds, 0.8, 'randomize');
%% Build a simple CNN 
imageSize = [227 227 3];
% Specify the convolutional neural network architecture.
layers = [
    imageInputLayer(imageSize)
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];
%% Specify training options 
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidationSet, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');
%% PART 2: Baseline Classifier with Data Augmentation
%% Create augmented image data store
% Specify data augmentation options and values/ranges
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5]);
% Apply transformations (using randomly picked values) and build augmented
% data store
augImds = augmentedImageDatastore(imageSize,imdsTrainingSet, ...
    'DataAugmentation',imageAugmenter);
% (OPTIONAL) Preview augmentation results 
batchedData = preview(augImds);
figure, imshow(imtile(batchedData.input))
    
%% Train the network. 
net = trainNetwork(augImds,layers,options);
analyzeNetwork(net)

%% Report accuracy of baseline classifier with image data augmentation
YPred = classify(net,imdsValidationSet);
YValidation = imdsValidationSet.Labels;
augImdsAccuracy = sum(YPred == YValidation)/numel(YValidation);
%% Plot confusion matrix
figure, plotconfusion(YValidation,YPred)