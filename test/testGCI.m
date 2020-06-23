nodeNum = 8;  % node number
sigLen = 200; % signal length
p = 3; % GC lag param

% generate random signals
%X = rand(nodeNum, sigLen); 

load('test/testTrain-rand500-uniform.mat');
X = si(1:8, 1:sigLen);

% set signal time lag 6->2, 6->4
X(2,3:end) = X(6,2:sigLen-1);
X(4,2:end) = X(6,1:sigLen-1);

%X(2,2:end) = X(6,1:sigLen-1);
%X(4,3:end) = X(2,2:sigLen-1);

gcI = calcPairwiseGCI(X, p); % calc granger causality index of lag |p|

% plot matrix
figure;
clims = [-10 10];
imagesc(gcI,clims);
title('pairwise Granger Causality Index');
colorbar;

gcI2 = calcMultivariateGCI(X, p); % calc granger causality index of lag |p|
gcI3 = calcMultivariateGCI2(X, p); % calc granger causality index of lag |p|

% plot matrix
figure;
clims = [-10 10];
imagesc(gcI2,clims);
title('multivariate Granger Causality Index');
colorbar;

% plot matrix
figure;
clims = [-10 10];
imagesc(gcI3,clims);
title('multivariate Granger Causality Index2');
colorbar;

