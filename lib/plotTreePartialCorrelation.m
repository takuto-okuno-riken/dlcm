%%
% Plot Tree Partial Correlation
% returns Tree Partial Correlation (PC)
% input:
%  X            multivariate time series matrix (node x time series)
%  exSignal     multivariate time series matrix (exogenous input x time series) (optional)
%  nodeControl  node control matrix (node x node) (optional)
%  exControl    exogenous input control matrix for each node (node x exogenous input) (optional)
%  isFullNode   return both node & exogenous causality matrix (optional)

function [PC] = plotTreePartialCorrelation(X, exSignal, nodeControl, exControl, isFullNode)
    if nargin < 5, isFullNode = 0; end
    if nargin < 4, exControl = []; end
    if nargin < 3, nodeControl = []; end
    if nargin < 2, exSignal = []; end

    PC = calcTreePartialCorrelation(X, exSignal, nodeControl, exControl, isFullNode);

    % show partial correlation
    clims = [-1,1];
    imagesc(PC,clims);
    daspect([1 1 1]);
    title(['Tree Partial Correlation']);
    xlabel('Source Nodes');
    ylabel('Target Nodes');
    colorbar;
end
