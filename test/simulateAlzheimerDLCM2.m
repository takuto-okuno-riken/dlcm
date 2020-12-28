% this function works after analyzeAlzheimerDLCM.m

function simulateAlzheimerDLCM
    % CONN fmri data base path :
    base = '../fmri/';

    % CONN output path
    pathesCN = {'ADNI2_65-78_F_CN_nii', 'ADNI2_65-78_M_CN_nii'};
    pathesAD = {'ADNI2_65-75_F_AD_nii', 'ADNI2_65-75_M_AD_nii'};

    % load each type signals
    [cnSignals, roiNames] = connData2signalsFile(base, pathesCN, 'cn');
    [adSignals] = connData2signalsFile(base, pathesAD, 'ad');

    cnSbjNum = length(cnSignals);
    adSbjNum = length(adSignals);
    nodeNum = size(cnSignals{1},1);
    
    % get DLCM-GC from original CN and AD (normal and recovery training)
    [cnDLs, meanCnDL, ~] = calculateConnectivity(cnSignals, roiNames, 'cn', 'dlcm', 1);
    [adDLs, meanAdDL, ~] = calculateConnectivity(adSignals, roiNames, 'ad', 'dlcm', 1);

    [rccnDLs, meanRcCnDL, ~] = calculateConnectivity(cnSignals, roiNames, 'cn', 'dlcmrc', 1); % do recovery training
    [rcadDLs, meanRcAdDL, ~] = calculateConnectivity(adSignals, roiNames, 'ad', 'dlcmrc', 1); % do recovery training

    % simulate CN & AD signals from first frame
    [cnDLWs, smcnSignals] = simulateNodeSignals(cnSignals, roiNames, 'cn', 'dlw', 'cn');
    [adDLWs, smadSignals] = simulateNodeSignals(adSignals, roiNames, 'ad', 'dlw', 'ad');

    % simulate recovery trained CN & AD signals from first frame
    [rccnDLWs, smrccnSignals] = simulateNodeSignals(cnSignals, roiNames, 'cn', 'dlwrc', 'cn');
    [rcadDLWs, smrcadSignals] = simulateNodeSignals(adSignals, roiNames, 'ad', 'dlwrc', 'ad');

    % simulate CN & AD signals from last-1 frame
    cn2Signals = cnSignals;
    ad2Signals = adSignals;
    for i=1:cnSbjNum
        cn2Signals{i}(:,1) = cn2Signals{i}(:,end-1);
        cn2Signals{i}(:,2) = cn2Signals{i}(:,end);
    end
    for i=1:adSbjNum
        ad2Signals{i}(:,1) = ad2Signals{i}(:,end-1);
        ad2Signals{i}(:,2) = ad2Signals{i}(:,end);
    end
    [~, smcn2Signals] = simulateNodeSignals(cn2Signals, roiNames, 'cn2', 'dlw', 'cn');
    [~, smad2Signals] = simulateNodeSignals(ad2Signals, roiNames, 'ad2', 'dlw', 'ad');

    % --------------------------------------------------------------------------------------------------------------
    % check DLCM-EC and DLCM-GC of simulated CN and AD
    [smcnDLs, meanSmcnDL, ~] = calculateConnectivity(smcnSignals, roiNames, 'smcn', 'dlcm', 1);
    [smcnDLWs, meanSmcnDLW, ~] = calculateConnectivity(smcnSignals, roiNames, 'smcn', 'dlw', 1);
    [smadDLs, meanSmadDL, ~] = calculateConnectivity(smadSignals, roiNames, 'smad', 'dlcm', 1);
    [smadDLWs, meanSmadDLW, ~] = calculateConnectivity(smadSignals, roiNames, 'smad', 'dlw', 1);

    [smrccnDLs, meanSmrccnDL, ~] = calculateConnectivity(smrccnSignals, roiNames, 'smrccn', 'dlcm', 1);
    [smrccnDLWs, meanSmrccnDLW, ~] = calculateConnectivity(smrccnSignals, roiNames, 'smrccn', 'dlw', 1);
    [smrcadDLs, meanSmrcadDL, ~] = calculateConnectivity(smrcadSignals, roiNames, 'smrcad', 'dlcm', 1);
    [smrcadDLWs, meanSmrcadDLW, ~] = calculateConnectivity(smrcadSignals, roiNames, 'smrcad', 'dlw', 1);
    
    [smcn2DLs, meanSmcn2DL, ~] = calculateConnectivity(smcn2Signals, roiNames, 'smcn2', 'dlcm', 1);
    [smcn2DLWs, meanSmcn2DLW, ~] = calculateConnectivity(smcn2Signals, roiNames, 'smcn2', 'dlw', 1);
    [smad2DLs, meanSmad2DL, ~] = calculateConnectivity(smad2Signals, roiNames, 'smad2', 'dlcm', 1);
    [smad2DLWs, meanSmad2DLW, ~] = calculateConnectivity(smad2Signals, roiNames, 'smad2', 'dlw', 1);
    
    meanCnDLW = nanmean(cnDLWs,3);
    meanAdDLW = nanmean(adDLWs,3);
    figure; cnsmcnDLWr = plotTwoSignalsCorrelation(meanCnDLW, meanSmcnDLW);
    figure; adsmadDLWr = plotTwoSignalsCorrelation(meanAdDLW, meanSmadDLW);
    figure; cnsmrccnDLWr = plotTwoSignalsCorrelation(meanCnDLW, meanSmrccnDLW);
    figure; adsmrcadDLWr = plotTwoSignalsCorrelation(meanAdDLW, meanSmrcadDLW);
    figure; cnsmcn2DLWr = plotTwoSignalsCorrelation(meanCnDLW, meanSmcn2DLW);
    figure; adsmad2DLWr = plotTwoSignalsCorrelation(meanAdDLW, meanSmad2DLW);

    % check FC of simulated CN and AD
    [cnFCs, meanCnFC, ~] = calculateConnectivity(cnSignals, roiNames, 'cn', 'fc', 1);
    [adFCs, meanAdFC, ~] = calculateConnectivity(adSignals, roiNames, 'ad', 'fc', 1);
    [smcnFCs, meanSmcnFC, ~] = calculateConnectivity(smcnSignals, roiNames, 'smcn', 'fc', 1);
    [smadFCs, meanSmadFC, ~] = calculateConnectivity(smadSignals, roiNames, 'smad', 'fc', 1);
    [smrccnFCs, meanSmrccnFC, ~] = calculateConnectivity(smrccnSignals, roiNames, 'smrccn', 'fc', 1);
    [smrcadFCs, meanSmrcadFC, ~] = calculateConnectivity(smrcadSignals, roiNames, 'smrcad', 'fc', 1);
    figure; cnsmcnFCr = plotTwoSignalsCorrelation(meanCnFC, meanSmcnFC);
    figure; cnsmrccnFCr = plotTwoSignalsCorrelation(meanCnFC, meanSmrccnFC);
    figure; adsmadFCr = plotTwoSignalsCorrelation(meanAdFC, meanSmadFC);
    figure; adsmrcadFCr = plotTwoSignalsCorrelation(meanAdFC, meanSmrcadFC);

    % plot box-and-whisker plot
    cosSims = nan(cnSbjNum,12);
    for i=1:cnSbjNum
        cosSims(i,1) = getCosSimilarity(cnDLWs(:,:,i), smcnDLWs(:,:,i));
        cosSims(i,3) = getCosSimilarity(cnDLs(:,:,i), smcnDLs(:,:,i));
        cosSims(i,5) = getCosSimilarity(cnDLWs(:,:,i), smrccnDLWs(:,:,i));
        cosSims(i,7) = getCosSimilarity(cnDLs(:,:,i), smrccnDLs(:,:,i));
        cosSims(i,9) = getCosSimilarity(cnFCs(:,:,i), smcnFCs(:,:,i));
        cosSims(i,11) = getCosSimilarity(cnFCs(:,:,i), smrccnFCs(:,:,i));
    end
    for i=1:adSbjNum
        cosSims(i,2) = getCosSimilarity(adDLWs(:,:,i), smadDLWs(:,:,i));
        cosSims(i,4) = getCosSimilarity(adDLs(:,:,i), smadDLs(:,:,i));
        cosSims(i,6) = getCosSimilarity(adDLWs(:,:,i), smrcadDLWs(:,:,i));
        cosSims(i,8) = getCosSimilarity(adDLs(:,:,i), smrcadDLs(:,:,i));
        cosSims(i,10) = getCosSimilarity(adFCs(:,:,i), smadFCs(:,:,i));
        cosSims(i,12) = getCosSimilarity(adFCs(:,:,i), smrcadFCs(:,:,i));
    end
    figure; boxplot(cosSims);
    
    % change Z score
%{
    cnDLWs = calcZScores(cnDLWs);
    adDLWs = calcZScores(adDLWs);
%}
    % plot correlation and cos similarity
    algNum = 20;
    cosSim = zeros(algNum,1);
    cosSim(1) = getCosSimilarity(meanCnDLW, meanSmcnDLW);
    cosSim(2) = getCosSimilarity(meanAdDLW, meanSmadDLW);
    cosSim(3) = getCosSimilarity(meanAdDLW, meanSmcnDLW);
    cosSim(4) = getCosSimilarity(meanCnDLW, meanSmadDLW);
    cosSim(5) = getCosSimilarity(meanCnDL, meanSmcnDL);
    cosSim(6) = getCosSimilarity(meanAdDL, meanSmadDL);
    cosSim(7) = getCosSimilarity(meanAdDL, meanSmcnDL);
    cosSim(8) = getCosSimilarity(meanCnDL, meanSmadDL);
    cosSim(9) = getCosSimilarity(meanCnFC, meanSmcnFC);
    cosSim(10) = getCosSimilarity(meanAdFC, meanSmadFC);
    cosSim(11) = getCosSimilarity(meanAdFC, meanSmcnFC);
    cosSim(12) = getCosSimilarity(meanCnFC, meanSmadFC);
    cosSim(13) = getCosSimilarity(meanCnDLW, meanSmrccnDLW);
    cosSim(14) = getCosSimilarity(meanAdDLW, meanSmrcadDLW);
    cosSim(15) = getCosSimilarity(meanAdDLW, meanSmrccnDLW);
    cosSim(16) = getCosSimilarity(meanCnDLW, meanSmrcadDLW);
    cosSim(17) = getCosSimilarity(meanCnFC, meanSmrccnFC);
    cosSim(18) = getCosSimilarity(meanAdFC, meanSmrcadFC);
    cosSim(19) = getCosSimilarity(meanAdFC, meanSmrccnFC);
    cosSim(20) = getCosSimilarity(meanCnFC, meanSmrcadFC);
    X = categorical({'cn-smcn','ad-smad','ad-smcn','cn-smad','cn-smcn-dlgc','ad-smad-dlgc','ad-smcn-dlgc','cn-smad-dlgc',...
        'cn-smcn-fc','ad-smad-fc','ad-smcn-fc','cn-smad-fc','cn-smrccn','ad-smrcad','ad-smrccn','cn-smrcad',...
        'cn-smrccn-fc','ad-smrcad-fc','ad-smrccn-fc','cn-smrcad-fc',});
    figure; bar(X, cosSim);
    title('cos similarity between CN and SimCN by each algorithm');

    % normality test
%{
    cnDLWsNt = calculateAlzNormalityTest(cnDLWs, roiNames, 'cnec', 'dlw');
%}
    % compalizon test (Wilcoxon, Mann?Whitney U test)
    [cnsmcnFCsUt, cnsmcnFCsUtP, cnsmcnFCsUtP2] = calculateAlzWilcoxonTest(cnFCs, smcnFCs, roiNames, 'cn', 'smcn', 'fc');
    [adsmadFCsUt, adsmadFCsUtP, adsmadFCsUtP2] = calculateAlzWilcoxonTest(adFCs, smadFCs, roiNames, 'ad', 'smad', 'fc');
    [cnsmrccnFCsUt, cnsmrccnFCsUtP, cnsmrccnFCsUtP2] = calculateAlzWilcoxonTest(cnFCs, smrccnFCs, roiNames, 'cn', 'smrccn', 'fc');
    [adsmrcadFCsUt, adsmrcadFCsUtP, adsmrcadFCsUtP2] = calculateAlzWilcoxonTest(adFCs, smrcadFCs, roiNames, 'ad', 'smrcad', 'fc');
    [cnsmcnDLsUt, cnsmcnDLsUtP, cnsmcnDLsUtP2] = calculateAlzWilcoxonTest(cnDLs, smcnDLs, roiNames, 'cn', 'smcn', 'dlcm');
    [adsmadDLsUt, adsmadDLsUtP, adsmadDLsUtP2] = calculateAlzWilcoxonTest(adDLs, smadDLs, roiNames, 'ad', 'smad', 'dlcm');
    [cnsmcnDLWsUt, cnsmcnDLWsUtP, cnsmcnDLWsUtP2] = calculateAlzWilcoxonTest(cnDLWs, smcnDLWs, roiNames, 'cn', 'smcn', 'dlw');
    [adsmadDLWsUt, adsmadDLWsUtP, adsmadDLWsUtP2] = calculateAlzWilcoxonTest(adDLWs, smadDLWs, roiNames, 'ad', 'smad', 'dlw');
    [cnsmrccnDLWsUt, cnsmrccnDLWsUtP, cnsmrccnDLWsUtP2] = calculateAlzWilcoxonTest(cnDLWs, smrccnDLWs, roiNames, 'cn', 'smrccn', 'dlw');
    [adsmrcadDLWsUt, adsmrcadDLWsUtP, adsmrcadDLWsUtP2] = calculateAlzWilcoxonTest(adDLWs, smrcadDLWs, roiNames, 'ad', 'smrcad', 'dlw');
end

% ==================================================================================================================

function [ECs, simSignals] = simulateNodeSignals(signals, roiNames, group, algorithm, orgGroup)
    % constant value
    ROINUM = size(signals{1},1);
    sbjNum = length(signals);

    outfName = ['results/adsim2-' algorithm '-' group '-roi' num2str(ROINUM) '.mat'];
    if exist(outfName, 'file')
        load(outfName);
        return;
    end

    % if you want to use parallel processing, set NumProcessors more than 2
    % and change for loop to parfor loop
    NumProcessors = 12;

    if NumProcessors > 1
        try
            disp('Destroing any existance matlab pool session');
            parpool('close');
        catch
            disp('No matlab pool session found');
        end
        parpool(NumProcessors);
    end
        
    ECs = zeros(ROINUM, ROINUM, sbjNum);
    simSignals = cell(1, sbjNum);
    parfor i=1:sbjNum    % for parallel processing
%    for i=1:sbjNum
        switch(algorithm)
        case {'dlw','dlwrc'}
            if strcmp(algorithm, 'dlw')
                dlcmName = ['results/ad-dlcm-' orgGroup '-roi' num2str(ROINUM) '-net' num2str(i) '.mat'];
            else
                dlcmName = ['results/ad-dlcmrc-' orgGroup '-roi' num2str(ROINUM) '-net' num2str(i) '.mat'];
            end
            f = load(dlcmName);
            [si, sig, c, maxsi, minsi] = convert2SigmoidSignal(signals{i});
            [Y, time] = simulateDlcmNetwork(si, f.inSignal, [], f.inControl, f.netDLCM);
            ec = calcDlcmEC(f.netDLCM, [], f.inControl);
        end
        ECs(:,:,i) = ec;
        simSignals{i} = Y;
    end
    save(outfName, 'ECs', 'simSignals', 'roiNames');

    % shutdown parallel processing
    if NumProcessors > 1
        delete(gcp('nocreate'))
    end
end


function [utestH, utestP, utestP2] = calculateAlzWilcoxonTest(control, target, roiNames, controlGroup, targetGroup, algorithm, force, testname, showfig)
    if nargin < 9
        showfig = 1;
    end
    if nargin < 8
        testname = 'ranksum';
    end
    if nargin < 7
        force = 0;
    end
    % constant value
    ROWNUM = size(control,1);
    COLNUM = size(control,2);

    outfName = ['results/adsim-' algorithm '-' controlGroup '_' targetGroup '-roi' num2str(ROWNUM) '-utest.mat'];
    if exist(outfName, 'file') && force == 0
        load(outfName);
    else
        utestH = nan(ROWNUM, COLNUM);
        utestP = nan(ROWNUM, COLNUM);
        utestP2 = nan(ROWNUM, COLNUM);
        for i=1:ROWNUM
            for j=1:COLNUM
                if isnan(control(i,j,1)), continue; end
                x = squeeze(control(i,j,:));
                y = squeeze(target(i,j,:));
                if length(x) > length(y)
                    x2 = x;
                    y2 = nan(length(x),1);
                    y2(1:length(y),1) = y;
                else
                    x2 = nan(length(y),1);
                    x2(1:length(x),1) = x;
                    y2 = y;
                end
                if isempty(find(~isnan(x2))) || isempty(find(~isnan(y2))), continue; end
                switch(testname)
                    case 'ranksum'
                        [p, h] = ranksum(x2,y2);
                    case 'ttest2'
                        [h, p] = ttest2(x2,y2);
                end
                %[p, h] = signrank(x2,y2);
                utestH(i,j) = h;
                utestP(i,j) = p;
                if h > 0 && nanmean(x) > nanmean(y)
                    utestP2(i,j) = p;
                end
            end
        end
        if force == 0
            save(outfName, 'utestH', 'utestP', 'utestP2', 'roiNames');
        end
    end
    % counting by source region and target region
    countSource = nansum(utestH,1);
    countTarget = nansum(utestH,2);
    save(outfName, 'utestH', 'utestP', 'utestP2', 'roiNames', 'countSource', 'countTarget');
    if showfig == 0, return; end
    
    load('test/colormap.mat')
    % U test result
    figure; 
    colormap(hvalmap);
    clims = [0,1];
    imagesc(utestH,clims);
    daspect([1 1 1]);
    title([controlGroup '-' targetGroup ' : ' algorithm ' : u test result']);
    colorbar;
    % U test p values
    utestP(isnan(utestP)) = 1;
    figure;
    colormap(pvalmap);
    clims = [0,1];
    imagesc(utestP, clims);
    daspect([1 1 1]);
    title([controlGroup '-' targetGroup ' : ' algorithm ' : u test p values']);
    colorbar;
end
