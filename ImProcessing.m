
% Load Data from folder and assigen data labels to folder names
imds = imageDatastore('VEIDD_Data', 'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Split data into 70% training and 30% testing 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.90,'randomized');

% Load pretrained netwrok
net = alexnet;

% Specify input (size extract from first layer)
inputSize = net.Layers(1).InputSize;
numClasses = numel(categories(imdsTrain.Labels));
% Replace Final Layers
layersTransfer = net.Layers(1:end-3);
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,...
    'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
% Create augmented set
pixelRange = [-40 40];
imageAugmenter = imageDataAugmenter( ...
    'RandScale',[0.8 1],...
    'RandRotation',[-50,50], ...    
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

% Resize all set to 227x227 pixels as required in first layer
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',16, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',0.001, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'Momentum', 0.9000);

% Train the new network based on transfer learning
netTransfer = trainNetwork(augimdsTrain,layers,options);

%Classify Validation Images
[YPred,scores] = classify(netTransfer,augimdsValidation);





