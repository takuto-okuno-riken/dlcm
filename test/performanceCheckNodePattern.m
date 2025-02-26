% Before using this function, download Dlingam-1.2 codes from
% https://sites.google.com/site/sshimizu06/Dlingamcode
% and add a path "Dlingam-1.2" and sub folders. And also download kernel-ICA 1.2 code from
% https://www.di.ens.fr/~fbach/kernel-ica/index.htm
% and add a path "kernel-ica1_2" and sub folders.

function performanceCheckNodePattern
    % load signals
    load('test/testTrain-rand500-uniform.mat');
    siOrg = si;
    
    nodeNum = 8;
    sigLen = 100;

    %% pattern 1 -------------------------------------------------
%%{
    disp('full random -- full independent nodes');
    si = siOrg(1:nodeNum,1:sigLen);
    checkingPattern(si, 1);
%%}
    %% pattern 2 -------------------------------------------------
%%{
    disp('node 2 and 6 are syncronized');
    si = siOrg(1:nodeNum, 1:sigLen);
    si(2,:) = si(6,:);
    checkingPattern(si, 2);
%%}
    %% pattern 3 -------------------------------------------------
%%{
    disp('node 2 is excited by node 6');
    si = siOrg(1:nodeNum, 1:sigLen);
    si(2,2:end) = si(6,1:sigLen-1);
    checkingPattern(si, 3);
%%}
    %% pattern 4 -------------------------------------------------
%%{
    disp('node 2 is excited half by node 6');
    si = siOrg(1:nodeNum, 1:sigLen);
    si(2,2:end) = si(6,1:sigLen-1) * 0.5;
    checkingPattern(si, 4);
%%}
    %% pattern 5 -------------------------------------------------
%%{
    disp('node 2,4 is excited by node 6');
    si = siOrg(1:nodeNum, 1:sigLen);
    si(2,2:end) = si(6,1:sigLen-1);
    si(4,2:end) = si(6,1:sigLen-1);
    checkingPattern(si, 5);
%%}
    %% pattern 6 -------------------------------------------------
%%{
    disp('nodes are excited 6-.->2, 2-.->4');
    si = siOrg(1:nodeNum, 1:sigLen);
    si(2,2:end) = si(6,1:sigLen-1);
    si(4,3:end) = si(2,2:sigLen-1);
    checkingPattern(si, 6);
%%}
    %% pattern # -------------------------------------------------
%{
    disp('node 2 and 6 are syncronized, but inverted');
    si = siOrg(1:nodeNum, 1:sigLen);
    si(2,:) = 1 - si(6,:);
    checkingPattern(si, 4);
%}
    %% pattern # -------------------------------------------------
%{
    disp('node 2 is inhibitted by node 6');
    si = siOrg(1:nodeNum, 1:sigLen);
    si(2,2:end) = 1 - si(6,1:sigLen-1);
    checkingPattern(si, 3);
%}
end

%% 
function [FC, DI, gcI] = checkingPattern(si, idx)
    nodeNum = size(si,1);
    sigLen = size(si,2);

    netFile = ['results/net-pat-' num2str(idx) '.mat'];
    if exist(netFile, 'file')
        load(netFile);
    else
        % layer parameters
        netDLCM = initMvarDnnNetwork(si);

        % show signals before training
        %{
        maxEpochs = 1;
        miniBatchSize = 1;
        options = trainingOptions('adam', ...
            'ExecutionEnvironment','cpu', ...
            'MaxEpochs',maxEpochs, ...
            'MiniBatchSize',miniBatchSize, ...
            'Shuffle','every-epoch', ...
            'GradientThreshold',5,...
            'Verbose',false);
    %            'Plots','training-progress');

        disp('initial state before training');
        netDLCM = trainMvarDnnNetwork(si, [], [], [], netDLCM, options);
        [t,mae,maeerr] = plotNodeSignals(nodeNum,si,exSignal,netDLCM);
        disp(['t=' num2str(t) ', mae=' num2str(mae)]);
        %}
        % training VARDNN network
        maxEpochs = 1000;
        miniBatchSize = ceil(sigLen / 3);
        options = trainingOptions('adam', ...
            'ExecutionEnvironment','cpu', ...
            'MaxEpochs',maxEpochs, ...
            'MiniBatchSize',miniBatchSize, ...
            'Shuffle','every-epoch', ...
            'GradientThreshold',5,...
            'Verbose',false);
    %            'Plots','training-progress');

        disp('start training');
        netDLCM = trainMvarDnnNetwork(si, [], [], [], netDLCM, options);  
        save(netFile, 'netDLCM');
    end

    % show signals after training
    figure; [S, t,mae,maeerr] = plotPredictSignals(si,[],[],[],netDLCM);
    disp(['t=' num2str(t) ', mae=' num2str(mae)]);

    % show original signal FC
    figure; FC = plotFunctionalConnectivity(si);
    figure; FCa = plotFunctionalConnectivityAbs(si); % calc FC (Abs)
    % show original signal PC
    figure; PC = plotPartialCorrelation(si);
    % show original signal mWCS
    figure; mWCS = plotWaveletCoherence(si);
    % show original signal granger causality index (GCI)
    figure; gcI = plotMultivariateGCI_(si, [], [], [], 3, 0);
    % show original time shifted correlation (tsc-FC)
    figure; tsCr = plotTimeShiftedCorrelation(si, [], [], [], 2);
    figure; tsCra = plotTimeShiftedCorrelationAbs(si, [], [], [], 2);
    % show deep-learning effective connectivity
%    figure; DI = plotMvarDnnECmeanWeight(netDLCM);
%    figure; DI = plotMvarDnnECmeanAbsWeight(netDLCM);
%    figure; DI = plotMvarDnnECmeanDeltaWeight(netDLCM);
%    figure; DI = plotMvarDnnECmeanAbsDeltaWeight(netDLCM);
    % show VARDNN-GC
    figure; dlGC = plotMvarDnnGCI(si, [], [], [], netDLCM, 0);
    % show VARDNN-WCI as VARDNN-DI
    figure; dlWC = plotMvarDnnDI(netDLCM, [], [], 0);
    % show DLCM-weight-GC
%    figure; dlwGC = plotDlcmDeltaWeightGCI(netDLCM);
    figure; Aest = plotDirectLiNGAM(si);
end

