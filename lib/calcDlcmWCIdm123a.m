%%
% get DLCM weight causality index matrix (wcI) and impaired node signals (wcNS)
% input:
%  netDLCM      trained DLCM network
%  nodeControl  node control matrix (node x node) (optional)
%  inControl    exogenous input control matrix for each node (node x exogenous input) (optional)
%  isFullNode   return both node & exogenous causality matrix (optional)

function [wcI, wcNS] = calcDlcmWCIdm123a(netDLCM, nodeControl, inControl, isFullNode)
    if nargin < 4
        isFullNode = 0;
    end
    if nargin < 3
        inControl = [];
    end
    if nargin < 2
        nodeControl = [];
    end
    nodeNum = length(netDLCM.nodeNetwork);
    nodeInNum = size(netDLCM.nodeNetwork{1, 1}.Layers(2, 1).Weights, 2);
    if isFullNode==0, nodeMax = nodeNum; else nodeMax = nodeInNum; end
    wcI = nan(nodeNum,nodeMax);
    wcNS = nan(nodeNum,nodeMax+1);
    for i=1:nodeNum
        % get input control
        control = ones(1, nodeNum);
        excontrol = ones(1, nodeInNum - nodeNum);
        if ~isempty(nodeControl)
            control = nodeControl(i,:);
        end
        if ~isempty(inControl)
            excontrol = inControl(i,:);
        end
        ctrl = [control, excontrol];

        % calc liner weights relation
        w1 = netDLCM.nodeNetwork{i, 1}.Layers(2, 1).Weights;
        w2 = netDLCM.nodeNetwork{i, 1}.Layers(4, 1).Weights;
        w3 = netDLCM.nodeNetwork{i, 1}.Layers(6, 1).Weights;
        b1 = netDLCM.nodeNetwork{i, 1}.Layers(2, 1).Bias;
        b2 = netDLCM.nodeNetwork{i, 1}.Layers(4, 1).Bias;
        b3 = netDLCM.nodeNetwork{i, 1}.Layers(6, 1).Bias;
        w1 = w1 .* repmat(ctrl, size(w1,1), 1);

        x = sum(w1,2) + b1;
        x(x<0) = 0;

        y = w2 * x + b2;
        y(y<0) = 0;
        z = w3 * y + b3;
        wcNS(i,1) = z;
        
        % imparement node signals
        for j=1:nodeMax
            if i==j, continue; end
            w = w1;
            w(:,j) = 0;
            xj = sum(w,2) + b1;
            xj(xj<0) = 0;
            
            yj = w2 * xj + b2;
            yj(yj<0) = 0;
            zj = w3 * yj + b3;
            wcI(i,j) = abs(z - zj);
            wcNS(i,j+1) = zj;
        end
    end
end