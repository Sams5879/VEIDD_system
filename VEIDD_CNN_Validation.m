
highscore = (int16(max(scores)*100));

% Extract category names & plot from test set


            
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    
    %title(strcat(string(label), string(highscore)));
    title(strcat(string(label)));
end
% Calculate the fraction of test images correctly classified by dividing numCorrect by the number of test images. 
%Store the result in a variable called fracCorrect.
restest = imdsValidation.Labels;
numCorrect = nnz(YPred == restest);
fracCorrect = numCorrect/numel(YPred);

% Confusion chart (incorrect prediction in testimages)
confusionchart(imdsValidation.Labels,YPred);