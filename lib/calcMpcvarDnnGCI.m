%%
% Caluclate multivariate PC VAR DNN Granger causality
% returns multivariate PC VAR DNN Granger causality index matrix (gcI), significance (h=1 or 0)
% p-values (P), F-statistic (F), the critical value from the F-distribution (cvFd)
% and AIC, BIC (of node vector)
% input:
%  X            multivariate time series matrix (node x time series)
%  exSignal     multivariate time series matrix (exogenous input x time series) (optional)
%  nodeControl  node control matrix (node x node) (optional)
%  exControl    exogenous input control matrix for each node (node x exogenous input) (optional)
%  net          trained multivariate VAR DNN network
%  alpha        the significance level of F-statistic (optional)
%  isFullNode   return both node & exogenous causality matrix (default:0)

function [gcI, h, P, F, cvFd, AIC, BIC, nodeAIC, nodeBIC] = calcMpcvarDnnGCI(X, exSignal, nodeControl, exControl, net, alpha, isFullNode)
    if nargin < 7, isFullNode = 0; end
    if nargin < 6, alpha = 0.05; end

    nodeNum = net.nodeNum;
    nodeInNum = nodeNum + net.exNum;
    sigLen = size(X,2);
    if isfield(net, 'lags'), lags = net.lags; else lags = 1; end
    if isFullNode==0, nodeMax = nodeNum; else nodeMax = nodeInNum; end
    p = net.lags;

    Y = [X; exSignal];
    Y = flipud(Y.'); % need to flip signal

    % first, calculate vector auto-regression (VAR) without target
    Yj = zeros(sigLen-p, p*nodeInNum);
    for k=1:p
        Yj(:,1+nodeInNum*(k-1):nodeInNum*k) = Y(1+k:sigLen-p+k,:);
    end

    % calc multivariate PC VAR DNN DI
    mu = net.mu;
    coeff = net.coeff;
    nodeNetwork = net.nodeNetwork;

    nodeAIC = zeros(nodeNum,1);
    nodeBIC = zeros(nodeNum,1);
    gcI = nan(nodeNum, nodeMax);
    h = nan(nodeNum,nodeMax);
    P = nan(nodeNum,nodeMax);
    F = nan(nodeNum,nodeMax);
    cvFd = nan(nodeNum,nodeMax);
    AIC = nan(nodeNum,nodeMax);
    BIC = nan(nodeNum,nodeMax);
    for i=1:nodeNum
        nodeIdx = [1:nodeNum];
        if ~isempty(nodeControl)
            [~,nodeIdx] = find(nodeControl(i,:)==1);
        end
        exIdx = [nodeNum+1:nodeInNum];
        if ~isempty(exControl)
            [~,exIdx] = find(exControl(i,:)==1);
            exIdx = exIdx + nodeNum;
        end
        idx = [];
        for k=1:p
            idx = [idx, nodeIdx+nodeInNum*(k-1), exIdx+nodeInNum*(k-1)];
        end
        nodeTeach = Y(1:sigLen-p,i);
        nodeInput = Yj(:,idx);
        
        Z = (nodeInput - repmat(mu{i},size(nodeInput,1),1)) / coeff{i}.';
        % predict 
        Si = predict(nodeNetwork{i}, Z.', 'ExecutionEnvironment', 'cpu');
        err = Si - nodeTeach.';
        VarEi = var(err,1);

        % AIC and BIC of this node (assuming residuals are gausiann distribution)
        T = sigLen-1;
        RSS = err*err';
        k = nodeNum + size(exSignal, 1) + 1; % input + bias
        %for j=2:2:length(nodeNetwork{i, 1}.Layers)
        %    k = k + length(nodeNetwork{i, 1}.Layers(j, 1).Bias);   % added hidden neuron number
        %end
        nodeAIC(i) = T*log(RSS/T) + 2 * k;
        nodeBIC(i) = T*log(RSS/T) + k*log(T);

        % imparement node signals
        for j=1:nodeMax
            if i==j, continue; end
            impInput = nodeInput;
            for p=1:lags, impInput(:,j+nodeInNum*(p-1)) = 0; end

            % predict 
            Z = (impInput - repmat(mu{i},size(nodeInput,1),1)) / coeff{i}.';
            Sj = predict(nodeNetwork{i}, Z.', 'ExecutionEnvironment', 'cpu');
            err = Sj - nodeTeach.';
            VarEj = var(err,1);
            gcI(i,j) = log(VarEj / VarEi);

            % AIC and BIC (assuming residuals are gausiann distribution)
            % BIC = n*ln(RSS/n)+k*ln(n)
            RSS1 = err*err';
            k1 = nodeNum - 1 + size(exSignal, 1) + 1;
            AIC(i,j) = T*log(RSS1/T) + 2 * k1;
            BIC(i,j) = T*log(RSS1/T) + k1*log(T);

            % calc F-statistic
            % https://en.wikipedia.org/wiki/F-test
            % F = ((RSS1 - RSS2) / (p2 - p1)) / (RSS2 / n - p2)
            %RSS1 = err*err';  % p1 = nodeNum - 1 + size(exSignal, 1) + 1;
            RSS2 = RSS;       % p2 = k
            F(i,j) = ((RSS1 - RSS2)/1) / (RSS2 / (sigLen - k));
            P(i,j) = 1 - fcdf(F(i,j),1,(sigLen - k));
            cvFd(i,j) = finv(1-alpha,1,(sigLen - k));
            h(i,j) = F(i,j) > cvFd(i,j);
        end
    end
end
