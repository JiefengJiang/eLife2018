%This function demonstrates key calculations of each model in the 
%model comparison of behavioral data
%
%Author: Jiefeng Jiang
%Ref: https://elifesciences.org/articles/39497
%
%
%Input arguments (all arrays, with each element representing a trial):
%pPrev: previous task (0 = motion, 1 = color)
%cue: probability of encountering a color task, as indicated by the precue
%mask: which trials are used for analysis (1 = yes, 0 = no). This is used
%to exclude error trials, etc
%task: current task encountered (0 = motion, 1 = color)
%RT: response time 
function BIC = ModelCrossValidation(pPrev, cue, mask, task, RT)

%assign 3 folds
fold1 = 1:150;
fold2 = 151:300;
fold3 = 301:450;

trainIdx = zeros(3, 450);
testIdx = zeros(3, 450);

trainIdx(1, [fold2 fold3]) = 1;
testIdx(1, fold1) = 1;

trainIdx(2, [fold1 fold3]) = 1;
testIdx(2, fold2) = 1;

trainIdx(3, [fold1 fold2]) = 1;
testIdx(3, fold3) = 1;

maskC = mask & (task > 0.5);
maskM = mask & (task < 0.5);

BIC = 0;

%After model comparison, all data was used together to determine the
%learning rate and the blending factor
for i = 1 : 3
    maskTrainingC = maskC & (trainIdx(i, :) > 0.5);
    maskTestC = maskC & (testIdx(i, :) > 0.5);
    maskTrainingM = maskM & (trainIdx(i, :) > 0.5);
    maskTestM = maskM & (testIdx(i, :) > 0.5);
    
    bestErr = -1;
    bestAlpha = -1;
    bestBC = [];
    bestBM = [];
    
    %for each fold, finding the modulation on PE
    
    %by fixing alpha to 1, you get PE_prev
    for alpha = 0:0.01:1
        preTask = pPrev;
        preTask([1:50:450]) = 0.5;
        for j = 2 : 450
            if mod(j, 50) ~= 1
                preTask(j) = preTask(j - 1) * (1 - alpha) + alpha * preTask(j);
            end
        end
        
        %by changing a to 1, you get results for PE_cue
        a = 0;
        prediction = preTask * (1 - a) + cue * a;

        %below you can add multiple PE vectors corresponding to the
        %different types of PEs above
        
        %Modeling color and motion tasks separately
        dmCTrain = [1 - prediction(maskTrainingC)]';
        dmCTrain = dmCTrain - mean(dmCTrain);
        yCTrain = RT(maskTrainingC)';
        yCTrain = yCTrain - mean(yCTrain);
        bC = pinv(dmCTrain) * yCTrain;

        dmMTrain = prediction(maskTrainingM)';
        dmMTrain = dmMTrain - mean(dmMTrain);
        yMTrain = RT(maskTrainingM)';
        yMTrain = yMTrain - mean(yMTrain);
        bM = pinv(dmMTrain) * yMTrain;

        err = [yCTrain - dmCTrain * bC; yMTrain - dmMTrain * bM];
        err = sum(err .^ 2);

        if (bestErr < 0 || bestErr > err)
            bestErr = err;
            bestAlpha = alpha;
            bestBC = bC;
            bestBM = bM;
        end
        
    end
    
    %Apply the modulation coefficients to the test fold and get prediction
    %errors
    
    preTask = pPrev;
    preTask([1:50:450]) = 0.5;
    for j = 2 : 450
        if mod(j, 50) ~= 1
            preTask(j) = preTask(j - 1) * (1 - bestAlpha) + bestAlpha * preTask(j);
        end
    end
    
    prediction = preTask * (1 - a) + cue * a;
    dmCTest = [1 - prediction(maskTestC)]';
    dmCTest = dmCTest - mean(dmCTest);
    yCTest = RT(maskTestC)';
    yCTest = yCTest - mean(yCTest);
    
    dmMTest = [prediction(maskTestM)]';
    dmMTest = dmMTest - mean(dmMTest);
    yMTest = RT(maskTestM)';
    yMTest = yMTest - mean(yMTest);
    
    err = [yCTest - dmCTest * bestBC; yMTest - dmMTest * bestBM];
    BIC = BIC + sum(err .^ 2);
end

n = sum(mask);
%the individual BICs for each model eventually go to SPM's spm_BMS function
%for model comparison
BIC = n * log(BIC / n);