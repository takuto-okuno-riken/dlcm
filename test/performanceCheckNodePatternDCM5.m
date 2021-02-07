% Before using this function, download SPM12 codes from
% https://www.fil.ion.ucl.ac.uk/spm/software/download/
% and add a path "spm12" and sub folders, then remove "spm12/external" folder and sub folders.

function performanceCheckNodePatternDCM5
    % set global random stream and shuffle it
    myStream=RandStream('mt19937ar');
    RandStream.setGlobalStream(myStream);
    rng('shuffle');

    % DEM Structure: create random inputs
    % -------------------------------------------------------------------------
    N  = 8;
    T  = 300;                             % number of observations (scans)
    TR = 2;                               % repetition time or timing
    n  = 8;                               % number of regions or nodes
    t  = (1:T)*TR;                        % observation times

    % priors
    % -------------------------------------------------------------------------
    options.maxnodes   = 4;  % effective number of nodes, 4 is better than n

    options.nonlinear  = 0;
    options.two_state  = 0;
    options.stochastic = 0;
    options.centre     = 1;
    options.induced    = 1;

    A   = eye(n,n);
    B   = zeros(n,n,0);
    C   = zeros(n,n);
    D   = zeros(n,n,0);
    pP  = spm_dcm_fmri_priors(A,B,C,D,options);

    pP.C = eye(n,n);
    pP.transit = randn(n,1)/16;

    % integrate states
    % -------------------------------------------------------------------------
    U.dt = TR;
    M.f  = 'spm_fx_fmri';
    M.x  = sparse(n,5);
    M.g   = 'spm_gx_fmri';

    %% pattern 1 -------------------------------------------------
%%{
    disp('network density 0.2'); % self connection
    pP.A = eye(n,n) * 0.2;
    pP.A(5,1) = 0.2 + rand() * 0.3;
    pP.A(8,3) = 0.2 + rand() * 0.3;
    pP.A(6,3) = 0.2 + rand() * 0.3;
    checkingPattern(pP,M,U,N,T,n,TR,options,1);
%%}
    %% pattern 2 -------------------------------------------------
%%{
    disp('network density 0.25');
    pP.A = eye(n,n) * 0.2;
    pP.A(5,1) = 0.3 + rand() * 0.3;
    pP.A(8,3) = 0.3 + rand() * 0.3;
    pP.A(6,3) = 0.3 + rand() * 0.3;
    pP.A(4,6) = 0.3 + rand() * 0.3;
    pP.A(7,6) = 0.3 + rand() * 0.3;
    pP.A(4,8) = 0.3 + rand() * 0.3;
    checkingPattern(pP,M,U,N,T,n,TR,options,2);
%%}
    %% pattern 6 -------------------------------------------------
%%{
    disp('network density 0.3');
    pP.A = eye(n,n) * 0.2;
    pP.A = addPattern6(pP.A,0.3,0.2);
    checkingPattern(pP,M,U,N,T,n,TR,options,6);
%%}
    %% pattern 7 -------------------------------------------------
%%{
    disp('network density 0.41');
    pP.A = eye(n,n) * 0.15;
    pP.A = addPattern6(pP.A,0.3,0.2);
    pP.A = addPattern7(pP.A);
    checkingPattern(pP,M,U,N,T,n,TR,options,7);
%%}
    %% pattern 8 -------------------------------------------------
%%{
    disp('network density 0.5');
    pP.A = eye(n,n) * 0.1;
    pP.A = addPattern6(pP.A,0.2,0.2);
    pP.A = addPattern7(pP.A);
    pP.A = addPattern8(pP.A);
    checkingPattern(pP,M,U,N,T,n,TR,options,8);
%%}
end

function A = addPattern6(A,base,range)
    A(3,1) = base + rand() * range;
    A(5,1) = base + rand() * range;
    A(8,3) = base + rand() * range;
    A(8,5) = base + rand() * range;
    A(5,8) = base + rand() * range;
    A(7,4) = base + rand() * range;
    A(6,7) = base + rand() * range;
    A(7,6) = base + rand() * range;
    A(6,8) = base + rand() * range;
end
function A = addPattern7(A)
    A(1,4) = 0.1 + rand() * 0.2;
    A(3,4) = 0.1 + rand() * 0.2;
    A(5,4) = 0.1 + rand() * 0.2;
    A(6,4) = 0.1 + rand() * 0.2;
    A(3,5) = 0.1 + rand() * 0.2;
    A(6,5) = 0.1 + rand() * 0.2;
end
function A = addPattern8(A)
    A(2,4) = 0.1 + rand() * 0.2;
    A(2,5) = 0.1 + rand() * 0.2;
    A(2,6) = 0.1 + rand() * 0.2;
    A(2,7) = 0.1 + rand() * 0.2;
    A(2,8) = 0.1 + rand() * 0.2;
end
function A = addPatternD(A,n)
    A(2,:) = NaN;
    A(:,2) = NaN;
    didx = find(eye(n,n)>0);
    A(didx) = NaN;
    idx = find(A>0);
    len = length(idx);
    idx = idx(randperm(len));
    A(idx(1:len-20)) = 0;
    idx = find(isnan(A));
    A(idx) = 0;
    A = A + eye(n,n) * (0.2);
end

%% 
function [FC, dlGC, gcI] = checkingPattern(pP,M,U,N,T,n,TR,options,idx)
    % show original connection
    figure; plotDcmEC(pP.A);
    maxLag = 5;

    fname = ['results/net-pat5-'  num2str(n) 'x' num2str(T) '-idx' num2str(idx) 'result.mat'];
    if exist(fname, 'file')
        load(fname);
    else
        dlAUC = zeros(maxLag,N);
        dlwAUC = zeros(maxLag,N);
        dl2AUC = zeros(maxLag,N);
        dlw2AUC = zeros(maxLag,N);
        for lags=1:maxLag
            dlROC{lags} = cell(N,2);
            dlwROC{lags} = cell(N,2);
            dl2ROC{lags} = cell(N,2);
            dlw2ROC{lags} = cell(N,2);
            dlRf{lags} = figure;
            dlwRf{lags} = figure;
            dl2Rf{lags} = figure;
            dlw2Rf{lags} = figure;
        end

        % calc input signal and node BOLD signals
        for k=1:N
            % read same y2, u2 si signals
            pat3File = ['results/net-pat3-'  num2str(n) 'x' num2str(T) '-idx' num2str(idx) '-' num2str(k) '.mat'];
            load(pat3File);

            % check saved data
            dlcmFile = ['results/net-pat5-'  num2str(n) 'x' num2str(T) '-idx' num2str(idx) '-' num2str(k) '.mat'];
            netDLCM = cell(maxLag,1);
            netDLCM2 = cell(maxLag,1);
            if exist(dlcmFile, 'file')
                load(dlcmFile);
            end
            % show DCM signals
            [si, sig, c, maxsi, minsi] = convert2SigmoidSignal(y2.', 0);
            [exSignal, sig2, c2, maxsi2, minsi2] = convert2SigmoidSignal(u2.', 0);
            exControl = eye(n,n);
            figure; plot(si.');
            %figure; plot(exSignal.');

            % training option
            sigLen = size(si,2);
            maxEpochs = 1000;
            miniBatchSize = ceil(sigLen / 3);
            options = trainingOptions('adam', ...
                'ExecutionEnvironment','cpu', ...
                'MaxEpochs',maxEpochs, ...
                'MiniBatchSize',miniBatchSize, ...
                'Shuffle','every-epoch', ...
                'GradientThreshold',5,...
                'L2Regularization',0.05, ...
                'Verbose',false);

            % train DLCM with lags
            for lags=1:maxLag
                if isempty(netDLCM{lags})
                    % train DLCM with normal activation function (ReLU)
                    netDLCM{lags} = initDlcmNetwork(si, exSignal, [], exControl, lags);
                    netDLCM{lags} = trainDlcmNetwork(si, exSignal, [], exControl, netDLCM{lags}, options);
                    save(dlcmFile, 'netDLCM', 'netDLCM2', 'pP', 'M', 'U','n','TR', 'y2', 'u2', 'si', 'data', 'sig', 'c', 'maxsi', 'minsi', 'sig2', 'c2', 'maxsi2', 'minsi2');
                end
                if isempty(netDLCM2{lags})
                    % train DLCM without activation function (ReLU) (linear case)
                    netDLCM2{lags} = initDlcmNetwork(si, exSignal, [], exControl, lags, []);
                    netDLCM2{lags} = trainDlcmNetwork(si, exSignal, [], exControl, netDLCM2{lags}, options);
                    save(dlcmFile, 'netDLCM', 'netDLCM2', 'pP', 'M', 'U','n','TR', 'y2', 'u2', 'si', 'data', 'sig', 'c', 'maxsi', 'minsi', 'sig2', 'c2', 'maxsi2', 'minsi2');
                end

                % show result of DLCM-GC
                dlGC = calcDlcmGCI(si, exSignal, [], exControl, netDLCM{lags}, 0);
                figure(dlRf{lags}); hold on; [dlROC{lags}{k,1}, dlROC{lags}{k,2}, dlAUC(lags,k)] = plotROCcurve(dlGC, pP.A); hold off;
                title(['DLCM(' num2str(lags) ')-GC']);

                % show result of DLCM-EC
                dlwGC = calcDlcmEC(netDLCM{lags}, [], exControl, 0);
                figure(dlwRf{lags}); hold on; [dlwROC{lags}{k,1}, dlwROC{lags}{k,2}, dlwAUC(lags,k)] = plotROCcurve(dlwGC, pP.A); hold off;
                title(['DLCM(' num2str(lags) ')-EC']);

                % show result of linear DLCM-GC
                dl2GC = calcDlcmGCI(si, exSignal, [], exControl, netDLCM2{lags}, 0);
                figure(dl2Rf{lags}); hold on; [dl2ROC{lags}{k,1}, dl2ROC{lags}{k,2}, dl2AUC(lags,k)] = plotROCcurve(dl2GC, pP.A); hold off;
                title(['linear DLCM(' num2str(lags) ')-GC']);

                % show result of linear DLCM EC 
                dlw2GC = calcDlcmEC(netDLCM2{lags}, [], exControl, 0);
                figure(dlw2Rf{lags}); hold on; [dlw2ROC{lags}{k,1}, dlw2ROC{lags}{k,2}, dlw2AUC(lags,k)] = plotROCcurve(dlw2GC, pP.A); hold off;
                title(['linear DLCM(' num2str(lags) ')-EC']);            
            end
        end
        save(fname, 'dlAUC', 'dlwAUC', 'dlROC', 'dlwROC', 'dl2AUC', 'dlw2AUC', 'dl2ROC', 'dlw2ROC');
    end

    % show average ROC curve of DCM
    figure; 
    hold on;
    for lags=1:maxLag
        plotErrorROCcurve(dlROC{lags}, N, [0.2,0.2,0.2]+(lags*0.1));
        plotErrorROCcurve(dlwROC{lags}, N, [0.2,0.2,0.2]+(lags*0.1));
        plotErrorROCcurve(dl2ROC{lags}, N, [0.2,0.2,0.3]+(lags*0.1));
        plotErrorROCcurve(dlw2ROC{lags}, N, [0.2,0.2,0.3]+(lags*0.1));
        plotAverageROCcurve(dlROC{lags}, N, '--', [0.2,0.2,0.2]+(lags*0.1),1.0);
        plotAverageROCcurve(dlwROC{lags}, N, '-', [0.2,0.2,0.2]+(lags*0.1),1.0);
        plotAverageROCcurve(dl2ROC{lags}, N, '--', [0.2,0.2,0.3]+(lags*0.1),0.4);
        plotAverageROCcurve(dlw2ROC{lags}, N, '-', [0.2,0.2,0.3]+(lags*0.1),0.4);
    end
    plot([0 1], [0 1],':','Color',[0.5 0.5 0.5]);
    hold off;
    ylim([0 1]);
    xlim([0 1]);
    daspect([1 1 1]);
    title(['averaged ROC curve idx' num2str(idx)]);
    xlabel('False Positive Rate')
    ylabel('True Positive Rate')
end

