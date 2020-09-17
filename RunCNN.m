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


load gregnet

YPred = classify(gregnet,imdsValidationSet);
YValidation = imdsValidationSet.Labels;
netTransfer1BaselineAccuracy = sum(YPred == YValidation)/numel(YValidation);
figure, plotconfusion(YValidation,YPred)


inputSize = gregnet.Layers(1).InputSize;

% Create augmentedImageDatastore from training and test sets to resize
% images in imds to the size required by the network.

augmentedTrainingSet = augmentedImageDatastore(inputSize, imdsTrainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(inputSize, imdsValidationSet, 'ColorPreprocessing', 'gray2rgb');



% Get the network weights for the second convolutional layer
w1 = gregnet.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights')


featureLayer = 'maxpool_2';
trainingFeatures = activations(gregnet, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');




% Get training labels from the trainingSet
trainingLabels = imdsTrainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');



% Extract test features using the CNN
testFeatures = activations(gregnet, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Get the known labels
testLabels = imdsValidationSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);
figure, plotconfusion(testLabels, predictedLabels)

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));



% Display the mean accuracy
mean(diag(confMat))







% Have user browse for a file, from a specified "starting folder."
% For convenience in browsing, set a starting folder from which to browse.
startingFolder = 'fundus';
if ~exist(startingFolder, 'dir')
  % If that folder doesn't exist, just start in the current folder.
  startingFolder = pwd;
end
% Get the name of the file that the user wants to use.
defaultFileName = fullfile(startingFolder, '*.jpg');
[baseFileName, folder] = uigetfile(defaultFileName, 'Select a file');
if baseFileName == 0
  % User clicked the Cancel button.
  return;
end
fullFileName = fullfile(folder, baseFileName);
testImage = imread(fullFileName);



% Create augmentedImageDatastore to automatically resize the image when
% image features are extracted using activations.
ds = augmentedImageDatastore(imageSize, testImage, 'ColorPreprocessing', 'gray2rgb');

% Extract image features using the CNN
imageFeatures = activations(gregnet, ds, featureLayer, 'OutputAs', 'columns');


% Make a prediction using the classifier
predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');

YPred = classify(gregnet,imdsValidationSet);
YValidation = imdsValidationSet.Labels;
imdsAccuracy = sum(YPred == YValidation)/numel(YValidation);
disp(predictedLabel)

