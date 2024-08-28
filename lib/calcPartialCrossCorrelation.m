%%
% Calculate Normalized Partial Cross-Correlation
% returns Normalized Partial Cross-Correlation (NPCC)
% input:
%  X            multivariate time series matrix (node x time series)
%  exSignal     multivariate time series matrix (exogenous input x time series) (optional)
%  nodeControl  node control matrix (node x node) (optional)
%  exControl    exogenous input control matrix for each node (node x exogenous input) (optional)
%  maxlag       maxlag of normalized cross-correlation [-maxlag, maxlag] (default:5)
%  isFullNode   return both node & exogenous causality matrix (optional)
%  lambda       ridge regression constant param (default:0)
%  usegpu       use gpu calculation (default:false)

function [NPCC, lags] = calcPartialCrossCorrelation(X, exSignal, nodeControl, exControl, maxlag, isFullNode, lambda, usegpu)
    if nargin < 8, usegpu = false; end
    if nargin < 7, lambda = 0; end
    if nargin < 6, isFullNode = 0; end
    if nargin < 5, maxlag = 5; end
    if nargin < 4, exControl = []; end
    if nargin < 3, nodeControl = []; end
    if nargin < 2, exSignal = []; end
    nodeNum = size(X,1);
    sigLen = size(X,2);
    exNum = size(exSignal,1);
    nodeMax = nodeNum + exNum;

    % set node input
    Y = [X; exSignal];
    if usegpu
        Y = gpuArray(single(Y));
    end

    % check all same value or not
    for i=1:nodeMax
        Ulen(i) = length(unique(single(Y(i,:)))); % 'half' does not support
    end
    
    NPCC = nan(nodeNum,nodeMax,maxlag*2+1,class(X));
    fullIdx = 1:nodeMax;
%    for i=1:nodeNum
    parfor i=1:nodeNum
        if ~isempty(nodeControl), nidx = find(nodeControl(i,:)==0); else nidx = []; end
        if ~isempty(exControl), eidx = find(exControl(i,:)==0); else eidx = []; end
        if ~isempty(eidx), eidx = eidx + nodeNum; end
        nodeIdx = setdiff(fullIdx,[nidx, eidx, i]);
        
        x = Y(i,:).';
        A = nan(nodeMax,maxlag*2+1,class(X));
        if usegpu
            A = gpuArray(single(A));
        end
        for j=i:nodeMax
            if j<=nodeNum && ~isempty(nodeControl) && nodeControl(i,j) == 0, continue; end
            if j>nodeNum && ~isempty(exControl) && exControl(i,j-nodeNum) == 0, continue; end
            
            if Ulen(i)==1 && Ulen(j)==1
                A(j,:) = 0;
            else
                idx = setdiff(nodeIdx,j);
                if lambda > 0
                    % MATLAB compatible version
                    z = Y(idx,:).';
                    y = Y(j,:).';
                    n = size(z,1);
                    mz = mean(z);
                    stdx = std(z,0,1);
                    MX = mz(ones(n,1),:);
                    STDX = stdx(ones(n,1),:);
                    Z = (z - MX) ./ STDX;

                    b1 = []; b2 = [];
                    zr = [Z; sqrt(lambda) * eye(size(Z,2))];
                    [~, ~, perm, RiQ] = regressPrepare(zr);
                    b1(perm) = RiQ * [x; zeros(size(Z,2),1)]; % p x 1
                    b1 = b1' ./ repmat(stdx',1,1);
                    r1 = x - (mean(x)-mz*b1 + z*b1);
                    b2(perm) = RiQ * [y; zeros(size(Z,2),1)]; % p x 1
                    b2 = b2' ./ repmat(stdx',1,1);
                    r2 = y - (mean(y)-mz*b2 + z*b2);
                else
                    z = [Y(idx,:).', ones(sigLen,1)]; % intercept to be close to partialcorr()

                    [~, ~, perm, RiQ] = regressPrepare(z);
                    [~, r1] = regressLinear(x, z, [], [], perm, RiQ);
                    [~, r2] = regressLinear(Y(j,:).', z, [], [], perm, RiQ);
                end

                [A(j,:), ~] = xcov(r1,r2,maxlag,'normalized');
            end
        end
        if usegpu
            NPCC(i,:,:) = gather(A);
        else
            NPCC(i,:,:) = A;
        end
    end
    A = ones(nodeNum,'logical'); A = tril(A,-1);
    idx = find(A==1);
    for i=1:size(NPCC,3)
        B = NPCC(:,1:nodeNum,maxlag*2+2-i); C = B';
        B = NPCC(:,1:nodeNum,i);
        B(idx) = C(idx);
        NPCC(:,1:nodeNum,i) = B;
    end
    lags = -maxlag:maxlag;

    % output control
    NPCC = NPCC(1:nodeNum,:,:);
    if isFullNode == 0
        NPCC = NPCC(:,1:nodeNum,:);
    end
    if ~isempty(nodeControl)
        nodeControl=double(nodeControl); nodeControl(nodeControl==0) = nan;
        NPCC(:,1:nodeNum,:) = NPCC(:,1:nodeNum,:) .* nodeControl;
    end
    if ~isempty(exControl) && isFullNode > 0
        exControl=double(exControl); exControl(exControl==0) = nan;
        NPCC(:,nodeNum+1:end,:) = NPCC(:,nodeNum+1:end,:) .* exControl;
    end
end
