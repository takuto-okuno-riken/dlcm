%%
% Calculate Support Vector Partial Correlation
% returns Support Vector Partial Correlation (PC)
% input:
%  X            multivariate time series matrix (node x time series)
%  exSignal     multivariate time series matrix (exogenous input x time series) (optional)
%  nodeControl  node control matrix (node x node) (optional)
%  exControl    exogenous input control matrix for each node (node x exogenous input) (optional)
%  kernel          kernel for SVM (default:'linear', 'gaussian', 'rbf')
%  kernelScale     kernelScale for SVM (default:'auto', 1)
%  isFullNode   return both node & exogenous causality matrix (optional)

function [PC] = calcSvPartialCorrelation(X, exSignal, nodeControl, exControl, kernel, kernelScale, isFullNode)
    if nargin < 7, isFullNode = 0; end
    if nargin < 6, kernelScale = 'auto'; end
    if nargin < 5, kernel = 'linear'; end
    if nargin < 4, exControl = []; end
    if nargin < 3, nodeControl = []; end
    if nargin < 2, exSignal = []; end

    nodeNum = size(X,1);
    sigLen = size(X,2);
    exNum = size(exSignal,1);
    nodeMax = nodeNum + exNum;

    % set node input
    Y = [X; exSignal];

    % check all same value or not
    uLen = zeros(nodeMax,1);
    for i=1:nodeMax
        uLen(i) = length(unique(Y(i,:)));
    end
    
    % each pool have each own gloval value.
%   setSvPCCFlag(false); % init global value for 'for loop'
%%{
    p = gcp; % If no pool, create new
    parfor i=1:p.NumWorkers
        setSvPCFlag(false); % init global value for 'parfor'
    end
%%}

    fullIdx = 1:nodeMax;
    PC = nan(nodeNum,nodeMax);
%    for i=1:nodeNum
    parfor i=1:nodeNum
        if getSvPCFlag() == true, continue; end
        if ~isempty(nodeControl), nidx = find(nodeControl(i,:)==0); else nidx = []; end
        if ~isempty(exControl), eidx = find(exControl(i,:)==0); else eidx = []; end
        if ~isempty(eidx), eidx = eidx + nodeNum; end
        nodeIdx = setdiff(fullIdx,[nidx, eidx, i]);

        A = nan(nodeMax,1);
        for j=i:nodeMax
            if j<=nodeNum && ~isempty(nodeControl) && nodeControl(i,j) == 0, continue; end
            if j>nodeNum && ~isempty(exControl) && exControl(i,j-nodeNum) == 0, continue; end
            if getSvPCFlag() == true, continue; end

            if uLen(i)==1 && uLen(j)==1
                A(j) = 0;
            else
                x = Y(i,:).';
                y = Y(j,:).';
                idx = setdiff(nodeIdx,j);
                z = Y(idx,:).';

                mdl = fitrsvm(z,x,'KernelFunction',kernel,'KernelScale',kernelScale); %,'Standardize',true); % bias will be calcurated
                Si = predict(mdl, z);
                r1 = Si - x;

                % if r1 are all NaN, finish this function
                if length(x) == sum(isnan(r1))
                    setSvPCFlag(true); continue;
                end

                mdl = fitrsvm(z,y,'KernelFunction',kernel,'KernelScale',kernelScale); %,'Standardize',true); % bias will be calcurated
                Si = predict(mdl, z);
                r2 = Si - y;

                A(j) = (r1.'*r2) / (sqrt(r1.'*r1)*sqrt(r2.'*r2));
            end
        end
        PC(i,:) = A;
    end
    for i=1:nodeNum
        for j=i:nodeMax, PC(j,i) = PC(i,j); end
    end

    % output control
    PC = PC(1:nodeNum,:);
    if isFullNode == 0
        PC = PC(:,1:nodeNum);
    end
    if ~isempty(nodeControl)
        nodeControl=double(nodeControl); nodeControl(nodeControl==0) = nan;
        PC(:,1:nodeNum) = PC(:,1:nodeNum) .* nodeControl;
    end
    if ~isempty(exControl) && isFullNode > 0
        exControl=double(exControl); exControl(exControl==0) = nan;
        PC(:,nodeNum+1:end) = PC(:,nodeNum+1:end) .* exControl;
    end
end

function setSvPCFlag(val)
    global svPCFlag;
    svPCFlag = val;
%    disp(['set svPCFlag = ' num2str(val)]);
end

function val = getSvPCFlag()
    global svPCFlag;
    val = svPCFlag;
%    disp(['get svPCFlag = ' num2str(val)]);
end
