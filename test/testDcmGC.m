

function testDcmGC
    % set global random stream and shuffle it
    myStream=RandStream('mt19937ar');
    RandStream.setGlobalStream(myStream);
    rng('shuffle');

    % DEM Structure: create random inputs
    % -------------------------------------------------------------------------
    N  = 8;
    T  = 300;                             % number of observations (scans)
    TR = 2;                               % repetition time or timing
    n  = 6;                               % number of regions or nodes
    t  = (1:T)*TR;                        % observation times

    % priors
    % -------------------------------------------------------------------------
    options.maxnodes   = 4;  % effective number of nodes, 4 is better than n

    options.nonlinear  = 0;
    options.two_state  = 0;
    options.stochastic = 0;
    options.centre     = 1;
    options.induced    = 1;

    A   = ones(n,n);
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
%{
    disp('full random -- full independent nodes');
    pP.A = rand(n,n)/5 - 0.1;
    checkingPattern(pP,M,U,N,T,n,TR,options,1);
%}
    %% pattern 2 -------------------------------------------------
%{
    disp('node 2 and 6 are syncronized');
    pP.A = rand(n,n)/5 - 0.1;
    checkingPattern(pP,M,U,N,T,n,TR,options,2);
%}
    %% pattern 3 -------------------------------------------------
%%{
    disp('node 2 is excited by node 6');
    pP.A = rand(n,n)/5 - 0.1;
    pP.A(2,6) = 1;
    checkingPattern(pP,M,U,N,T,n,TR,options,3);
%%}
    %% pattern 4 -------------------------------------------------
%%{
    disp('node 2 is excited half by node 6');
    pP.A = rand(n,n)/5 - 0.1;
    pP.A(2,6) = 0.5;
    checkingPattern(pP,M,U,N,T,n,TR,options,4);
%%}
    %% pattern 5 -------------------------------------------------
%%{
    disp('node 2,4 is excited by node 6');
    pP.A = rand(n,n)/5 - 0.1;
    pP.A(2,6) = 1;
    pP.A(4,6) = 1;
    checkingPattern(pP,M,U,N,T,n,TR,options,5);
%%}
    %% pattern 6 -------------------------------------------------
%%{
    disp('nodes are excited 6->2, 2->4');
    pP.A = rand(n,n)/5 - 0.1;
    pP.A(2,6) = 1;
    pP.A(4,2) = 1;
    checkingPattern(pP,M,U,N,T,n,TR,options,6);
%%}
%{
    %% pattern 7 -------------------------------------------------
%%{
    disp('nodes are excited 6->2, 4->2');
    pP.A = rand(n,n)/5 - 0.1;
    pP.A(2,6) = 1;
    pP.A(2,4) = 1;
    checkingPattern(pP,M,U,N,T,n,TR,options,7);
%%}
    %% pattern 8 -------------------------------------------------
%%{
    disp('nodes are excited 6->2, 5->1');
    pP.A = rand(n,n)/5 - 0.1;
    pP.A(2,6) = 1;
    pP.A(1,5) = 1;
    checkingPattern(pP,M,U,N,T,n,TR,options,8);
%%}
    %% pattern 9 -------------------------------------------------
%%{
    disp('nodes are excited 6->2,6->1, 1->3,1->5, 5->4');
    pP.A = rand(n,n)/20 - 0.025;
    pP.A(2,6) = 0.6;
    pP.A(1,6) = 0.6;
    pP.A(3,1) = 0.6;
    pP.A(5,1) = 0.6;
    pP.A(4,5) = 0.6;
    checkingPattern(pP,M,U,N,T,n,TR,options,9);
%%}
%}
end

%% 
function [gcI] = checkingPattern(pP,M,U,N,T,n,TR,options,idx)
    dcmFile = ['results/dcm-gc-test' num2str(idx) '-' num2str(n) 'x' num2str(T) '.mat'];
    netDLCM = [];

    % show original connection
    figure; plotDcmEC(pP.A);
    
    if exist(dcmFile, 'file')
        load(dcmFile);
    else
        CSD = {};
        RMS = [];
        Uus = {};
    end

    % initialize DCM stcuct
    DCM = struct();
    DCM.options = options;

    DCM.a    = ones(n,n);
    DCM.b    = zeros(n,n,0);
    DCM.c    = eye(n,n);
    DCM.d    = zeros(n,n,0);

    DCM.Y.dt = TR;
    DCM.U.dt = TR;

    % performance check of DCM inversion
    for k=length(CSD)+1:N
        % generate signal by DCM
        U.u = spm_rand_mar(T+50,n,1/2)/8;       % endogenous fluctuations
        y   = spm_int_J(pP,M,U);                % integrate with observer
        y2  = y(51:end,:);
        u2  = U.u(51:end,:);
        si = y2.';
        Uus{end + 1} = U.u;

        % response
        % -----------------------------------------------------------------
        DCM.Y.y  = y2;
        DCM.U.u  = u2;

        % nonlinear system identification (Variational Laplace)
        % =================================================================
        CSD{end + 1} = spm_dcm_fmri_csd(DCM);
        BPA          = spm_dcm_average(CSD,'simulation',1);

        dp   = BPA.Ep.A - pP.A;
        dp   = dp - diag(diag(dp));
        RMS(end + 1) = sqrt(mean(dp(~~dp).^2))

        A = BPA.Ep.A;
    end
    save(dcmFile, 'netDLCM', 'pP', 'M', 'U','n','TR', 'y2', 'u2', 'si', 'A', 'Uus', 'RMS', 'CSD');

    % check DCM gc
    orgY = spm_int_J(pP,M,Uus{end});                % integrate with observer
    invpP = pP;
    invpP.A = A;
    invY = spm_int_J(invpP,M,Uus{end});             % integrate with observer    
    invErr = invY - orgY;

    gcI = nan(n, n);
    for i=1:n
        VarEi = var(invErr(:,i));

        uuj = Uus{end};
        uuj(:,i) = 0;
        invpPj = invpP;
        invpPj.A(i,:) = 0;
        invpPj.A(:,i) = 0;
        % predict 
        invYj = spm_int_J(invpPj,M,uuj);             % integrate with observer    
        invErrj = invYj - orgY;
        % imparement node signals
        for j=1:n
            if i==j, continue; end
            VarEj = var(invErrj(:,j));
            gcI(i,j) = log(VarEj / VarEi);
        end
    end

    % show DCM GC
    figure;
    sigma = std(gcI(:),'omitnan');
    avg = mean(gcI(:),'omitnan');
    gcI = (gcI - avg) / sigma;
    range = 5;
    clims = [-range, range];
    imagesc(gcI,clims);
    daspect([1 1 1]);
    title('DCM Granger Causality Index (DCM-GC)');
    colorbar;

    % show GC
    figure; gcI = plotMultivariateGCI(orgY.',3,0);
end
