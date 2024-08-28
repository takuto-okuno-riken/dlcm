nodeNum = 8;  % node number
sigLen = 100; % signal length

% generate random signals
%X = rand(nodeNum, sigLen); 
load('test/testTrain-rand500-uniform.mat');
X = si(1:nodeNum, 1:sigLen);
x = X(1,:).';
y = X(2,:).';
z = X(3:nodeNum,:).';

% matlab imprementation
[PC, P] = partialcorr(X.');
pc1 = PC(1,2);
pc2 = partialcorr(x,y,z);
% regression version (original definition)
z1 = [z, ones(sigLen,1)];
[b1,bint1,r1] = regress(x,z1);
[b2,bint2,r2] = regress(y,z1);
pc3 = (r1.'*r2) / (sqrt(r1.'*r1)*sqrt(r2.'*r2));
pc4 = corr(r1,r2);
% invert of cov matrix version
sigma = cov(X.',1);
%a = sqrt(mean(diag(sigma).^2)); sigma = sigma / a; % normalize
B = inv(sigma);
pc5 = -B(1,2) / sqrt(B(1,1)*B(2,2));

% redge regression version
sigma = cov(X.',1);
lambda = 0.01;
B = inv(sigma + lambda*eye(size(sigma,1)));
pc6 = -B(1,2) / sqrt(B(1,1)*B(2,2));
%grot=-B;
%grot=(grot ./ repmat(sqrt(abs(diag(grot))),1,8)) ./ repmat(sqrt(abs(diag(grot)))',8,1);

k = 100;
B1 = ridge(x,z, k, 0);
r1 = x - (B1(1)+z*B1(2:end));
B2 = ridge(y,z, k, 0);
r2 = y - (B2(1)+z*B2(2:end));
pc7 = corr(r1,r2);

% redge regression (MATLAB compatible) version
[n,p] = size(z);
mz = mean(z);
stdx = std(z,0,1);
MX = mz(ones(n,1),:);
STDX = stdx(ones(n,1),:);
Z = (z - MX) ./ STDX;

zr = [Z; sqrt(k) * eye(size(Z,2))];
xr = [x; zeros(size(Z,2),1)];
yr = [y; zeros(size(Z,2),1)];
b1 = zr \ xr;
b1 = b1 ./ repmat(stdx',1,1);
r1 = x - (mean(x)-mz*b1 + z*b1);
b2 = zr \ yr;
b2 = b2 ./ repmat(stdx',1,1);
r2 = y - (mean(y)-mz*b2 + z*b2);
pc8 = corr(r1,r2);

[Q, R, perm, RiQ] = regressPrepare(zr);
b1(perm) = RiQ * xr;
b1 = b1 ./ repmat(stdx',1,1);
r1 = x - (mean(x)-mz*b1 + z*b1);
b2(perm) = RiQ * yr;
b2 = b2 ./ repmat(stdx',1,1);
r2 = y - (mean(y)-mz*b2 + z*b2);
pc9 = corr(r1,r2);


%% cos similarity original vs. sin, v09, mFT
%{
n = 8;
U = tril(nan(n));
load(['results/simsc/sin-' num2str(n) 'x2000-idx6-1-1-result.mat']);
plotTwoSignals(S(:,:,1),S(:,:,2),0,[0, 1]);
figure; [NC1] = plotPartialCorrelation(S(:,:,1), [], [], []);
figure; [NC2] = plotPartialCorrelation(S(:,:,2), [], [], []);
getCosSimilarity(NC1+U, NC2+U)
figure; [PNC1] = plotSvPartialCorrelation(S(:,:,1), [], [], [], 'gaussian');
figure; [PNC2] = plotSvPartialCorrelation(S(:,:,2), [], [], [], 'gaussian');
getCosSimilarity(PNC1+U, PNC2+U)
load(['results/simsc/v09-' num2str(n) 'x2000-idx6-1-1-result.mat']);
figure; [PNC2] = plotSvPartialCorrelation(S(:,:,2), [], [], [], 'gaussian');
getCosSimilarity(PNC1+U, PNC2+U)
load(['results/simsc/mft-' num2str(n) 'x2000-idx6-1-1-result.mat']);
figure; [PNC2] = plotSvPartialCorrelation(S(:,:,2), [], [], [], 'gaussian');
getCosSimilarity(PNC1+U, PNC2+U)
%}

%% test pattern 1 -- just random
% regression version
PC3 = calcPartialCorrelation__(X);
Z = PC - PC3;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC3 : sum err=' num2str(nansum(abs(Z),'all'))]);

% invert version
PC3b = calcPartialCorrelation_(X);
Z = PC - PC3b;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC3b : sum err=' num2str(nansum(abs(Z),'all'))]);

% zero-lag pcc regression version
PC3c = calcPartialCrossCorrelation(X,[],[],[],2);
Z = PC - PC3c(:,:,3);
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC3c : sum err=' num2str(nansum(abs(Z),'all'))]);

% redge regression invert version
PC3d = calcPartialCorrelation_(X,[],[],[],[],0.1);
Z = PC - PC3d;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC3d : sum err=' num2str(nansum(abs(Z),'all'))]);

PC3e = calcPartialCrossCorrelation(X,[],[],[],2,0,200);
Z = PC3d - PC3e(:,:,3);
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC3d - PC3e : sum err=' num2str(nansum(abs(Z),'all'))]);
nansum(abs(PC - PC3e(:,:,3)),'all')

PC4 = calcPLSPartialCorrelation(X); % calc PLS PC
Z = PC - PC4;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC4 : sum err=' num2str(nansum(abs(Z),'all'))]);

PC5 = calcLassoPartialCorrelation(X, [], [], [], 0.01, 1); % calc Lasso PC
Z = PC - PC5;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC5 : sum err=' num2str(nansum(abs(Z),'all'))]);

[lambda, alpha, errMat] = estimateLassoParamsForPC(X, [], [], [], 0.5, 5, [0.01:0.02:0.99],[1:-0.1:0.1]);
PC5b = calcLassoPartialCorrelation(X, [], [], [], lambda, alpha); % calc Lasso PC
Z = PC - PC5b; nansum(abs(Z),'all')

PC6 = calcPcPartialCorrelation(X); % calc PCA+PC
Z = PC - PC6;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC6 : sum err=' num2str(nansum(abs(Z),'all'))]);

PC7 = calcSvPartialCorrelation(X); % calc SVR(linear)+PC
Z = PC - PC7;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC7 : sum err=' num2str(nansum(abs(Z),'all'))]);

PC8 = calcSvPartialCorrelation(X,[],[],[],'gaussian'); % calc SVR(gaussian)+PC
Z = PC - PC8;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC8 : sum err=' num2str(nansum(abs(Z),'all'))]);

PC9 = calcSvPartialCorrelation(X,[],[],[],'rbf'); % calc SVR(rbf)+PC
Z = PC - PC9;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC9 : sum err=' num2str(nansum(abs(Z),'all'))]);

PC10 = calcGpPartialCorrelation(X); % calc GP+PC
Z = PC - PC10;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC10 : sum err=' num2str(nansum(abs(Z),'all'))]);

PC11 = calcTreePartialCorrelation(X); % calc Tree+PC
Z = PC - PC11;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC11 : sum err=' num2str(nansum(abs(Z),'all'))]);

PC12 = calcRfPartialCorrelation(X); % calc RF+PC
Z = PC - PC12;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC12 : sum err=' num2str(nansum(abs(Z),'all'))]);

% set training options
maxEpochs = 400;
sigLen = size(X,2);
miniBatchSize = ceil(sigLen / 3);

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'L2Regularization',0.05, ...
    'GradientThreshold',5,...
    'Verbose',false);

PC13 = calcDnnPartialCorrelation(X, [], [], [], options); % calc DNN+PC
Z = PC - PC13;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC13 : sum err=' num2str(nansum(abs(Z),'all'))]);

%% test pattern 2 -- one step
exNum = 2;
exSignal = si(nodeNum+1:nodeNum+exNum,1:sigLen);
% control is all positive input
exControl = ones(nodeNum,exNum);
X = si(1:8, 1:sigLen);
X(2,2:end) = X(6,1:sigLen-1);
X(4,2:end) = X(6,1:sigLen-1); % caution! node 2 & 4 is Multicollinearity case (correlated)
X(5,2:end) = exSignal(1,1:sigLen-1);
%X(7,2:end) = exSignal(1,1:sigLen-1);

figure; PC = plotPartialCorrelation(X, exSignal, [], exControl, 1); % calc PC - many NaN values
PC3 = calcPartialCorrelation__(X, exSignal, [], exControl, 1);
figure; PC4 = plotPLSPartialCorrelation(X, exSignal, [], exControl, 1); % calc PLS PC
figure; PC5 = plotLassoPartialCorrelation(X, exSignal, [], exControl, 0.01, 1, 1); % calc Lasso PC
figure; PC6 = plotPcPartialCorrelation(X, exSignal, [], exControl, 1, 1); % calc PCA+PC
figure; PC7 = plotSvPartialCorrelation(X, exSignal, [], exControl, 'linear', 'auto', 1); % calc SVR(linber)+PC
figure; PC8 = plotSvPartialCorrelation(X, exSignal, [], exControl, 'gaussian', 'auto', 1); % calc SVR(gaussian)+PC
figure; PC9 = plotSvPartialCorrelation(X, exSignal, [], exControl, 'rbf', 'auto', 1); % calc SVR(rbf)+PC
figure; PC10 = plotGpPartialCorrelation(X, exSignal, [], exControl, 'squaredexponential', 'linear', 1); % calc GP+PC
figure; PC11 = plotTreePartialCorrelation(X, exSignal, [], exControl, 1); % calc Tree+PC
figure; PC12 = plotRfPartialCorrelation(X, exSignal, [], exControl, 10, 1); % calc Rf+PC
figure; PC13 = plotDnnPartialCorrelation(X, exSignal, [], exControl, options, @reluLayer, 1); % calc DNN+PC

Z = PC - PC3;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC3 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC - PC4;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC4 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC - PC5;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC5 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC - PC6;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC6 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC - PC7;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC7 : sum err=' num2str(nansum(abs(Z),'all'))]);
Z = PC - PC8;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC8 : sum err=' num2str(nansum(abs(Z),'all'))]);
Z = PC - PC9;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC9 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC - PC10;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC10 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC - PC11;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC11 : sum err=' num2str(nansum(abs(Z),'all'))]);
Z = PC - PC12;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC12 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC - PC13;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC13 : sum err=' num2str(nansum(abs(Z),'all'))]);

%% test pattern 3 -- one step
exNum = 8;
exSignal = si(nodeNum+1:nodeNum+exNum,1:sigLen);
% control is all positive input
exControl = eye(nodeNum,exNum);
X = si(1:8, 1:sigLen);
X(2,2:end) = X(6,1:sigLen-1);
X(4,2:end) = X(6,1:sigLen-1); % caution! node 2 & 4 is Multicollinearity case (correlated)
X(5,2:end) = exSignal(5,1:sigLen-1);
X(7,2:end) = exSignal(7,1:sigLen-1);

figure; PC = plotPartialCorrelation(X, exSignal, [], exControl, 1); % calc PC - many NaN values
PC3 = calcPartialCorrelation__(X, exSignal, [], exControl, 1);
figure; PC4 = plotPLSPartialCorrelation(X, exSignal, [], exControl, 1); % calc PLS PC
figure; PC5 = plotLassoPartialCorrelation(X, exSignal, [], exControl, 0.01, 1, 1); % calc Lasso PC
figure; PC6 = plotPcPartialCorrelation(X, exSignal, [], exControl, 1, 1); % calc PCA+PC
figure; PC7 = plotSvPartialCorrelation(X, exSignal, [], exControl, 'linear', 'auto', 1); % calc SVR(linber)+PC
figure; PC8 = plotSvPartialCorrelation(X, exSignal, [], exControl, 'gaussian', 'auto', 1); % calc SVR(gaussian)+PC
figure; PC9 = plotSvPartialCorrelation(X, exSignal, [], exControl, 'rbf', 'auto', 1); % calc SVR(rbf)+PC
figure; PC10 = plotGpPartialCorrelation(X, exSignal, [], exControl, 'squaredexponential', 'linear', 1); % calc GP+PC
figure; PC11 = plotTreePartialCorrelation(X, exSignal, [], exControl, 1); % calc Tree+PC
figure; PC12 = plotRfPartialCorrelation(X, exSignal, [], exControl, 10, 1); % calc Rf+PC
figure; PC13 = plotDnnPartialCorrelation(X, exSignal, [], exControl, options, @reluLayer, 1); % calc DNN+PC

Z = PC - PC3;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC3 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC - PC4;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC4 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC - PC5;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC5 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC - PC6;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC6 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC - PC7;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC7 : sum err=' num2str(nansum(abs(Z),'all'))]);
Z = PC - PC8;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC8 : sum err=' num2str(nansum(abs(Z),'all'))]);
Z = PC - PC9;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC9 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC - PC10;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC10 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC - PC11;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC11 : sum err=' num2str(nansum(abs(Z),'all'))]);
Z = PC - PC12;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC12 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC - PC13;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC - PC13 : sum err=' num2str(nansum(abs(Z),'all'))]);

%% test pattern 4 -- rank down
% synchronize signal 6 == 2, 7 invert 3
% this is rank down operation, then invert of cov matrix does not work!!
X = si(1:8, 1:sigLen);
X(2,:) = X(6,:);
X(3,:) = 1 - X(7,:);

figure; PC = plotPartialCorrelation(X); % calc PC - many NaN values
PC2 = calcPartialCorrelation_(X,0); % calc PC - this does not work well.
PC3 = calcPartialCorrelation__(X); % calc PC
figure; PC4 = plotPLSPartialCorrelation(X); % calc PLS PC
figure; PC5 = plotLassoPartialCorrelation(X, [], [], [], 0.01, 1); % calc Lasso PC
figure; PC6 = plotPcPartialCorrelation(X, [], [], [], 1); % calc PCA+PC
figure; PC7 = plotSvPartialCorrelation(X, [], [], [], 'linear', 'auto', 1); % calc SVR(linber)+PC
figure; PC8 = plotSvPartialCorrelation(X, [], [], [], 'gaussian', 'auto', 1); % calc SVR(gaussian)+PC
figure; PC9 = plotSvPartialCorrelation(X, [], [], [], 'rbf', 'auto', 1); % calc SVR(rbf)+PC
figure; PC10 = plotGpPartialCorrelation(X, [], [], [], 'squaredexponential', 'linear', 1); % calc GP+PC
figure; PC11 = plotTreePartialCorrelation(X, [], [], [], 1); % calc Tree+PC
figure; PC12 = plotRfPartialCorrelation(X, [], [], [], 10, 1); % calc Rf+PC
figure; PC13 = plotDnnPartialCorrelation(X, [], [], [], options, @reluLayer, 1); % calc DNN+PC

[lambda, alpha, errMat] = estimateLassoParamsForPC(X, [], [], [], 0.5, 5, [0.01:0.02:0.99],[1:-0.1:0.1]);
PC5b = calcLassoPartialCorrelation(X, [], [], [], lambda, alpha); % calc Lasso PC
Z = PC3 - PC5b; nansum(abs(Z),'all')

% plot matrix
figure;
clims = [-1 1];
imagesc(PC2,clims);
title('Partial Correlation (invert of cov matrix)');
colorbar;

% plot matrix
figure;
clims = [-1 1];
imagesc(PC3,clims);
title('Partial Correlation (regression base)');
colorbar;

% plot matrix
Z = PC3 - PC4;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC3 - PC4 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC3 - PC5;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC3 - PC5 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC3 - PC6;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC3 - PC6 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC3 - PC7;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC3 - PC7 : sum err=' num2str(nansum(abs(Z),'all'))]);
Z = PC3 - PC8;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC3 - PC8 : sum err=' num2str(nansum(abs(Z),'all'))]);
Z = PC3 - PC9;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC3 - PC9 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC3 - PC10;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC3 - PC10 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC3 - PC11;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC3 - PC11 : sum err=' num2str(nansum(abs(Z),'all'))]);
Z = PC3 - PC12;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC3 - PC12 : sum err=' num2str(nansum(abs(Z),'all'))]);

Z = PC3 - PC13;
figure; clims = [-1 1]; imagesc(Z,clims); title(['PC3 - PC13 : sum err=' num2str(nansum(abs(Z),'all'))]);
