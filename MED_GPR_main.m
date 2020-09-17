%%%%%%%%%%      Gaussian Process Regression (GPR)               %%%%%%%%%
% Demo: prediction using GPR
% ---------------------------------------------------------------------%

clc
close all
clear all
addpath(genpath(pwd))


formatSpec = '%e ';% " " space seperates columns
sizeA = [12, inf];% structure of file
fileID = fopen('./data/4l_trial.txt', 'r');
% 4l_trial.txt has 12 columns, columns 1:8 are 1st till 8th moments,
% remaining 4 columns 9:12 are Lagrange multlipliers lambda_1,...lambda_4
A = fscanf(fileID,formatSpec, sizeA)';


Nt = 1000;% number of training points
Ns = 1000; % number of testing points
pt = A(1:Nt,3:4);% input moments, training
lat = A(1:Nt,9:12);% output lagrange multipliers, training


%ps = A(1:Nt,3:4);% input moments, testing same as training points
%las = A(1:Nt,8:12);% output lagrange multipliers, testing points same as training

ps = A(Nt+1:Nt+Ns,3:4);% input moments, testing points (untrained data)
las = A(Nt+1:Nt+Ns,9:12);% output lagrange multipliers, testing points (untrained data)

% Set the mean function, covariance function and likelihood function
% Take meanConst, covRQiso and likGauss as examples
meanfunc = @meanConst;
%covfunc = @covRQiso; 
covfunc = @covSEard;
likfunc = @likGauss; 

% Initialization of hyperparameters
%hyp = struct('mean', 3, 'cov', [2 2 2], 'lik', -1);
hyp = struct('mean', 0, 'cov', [.1 1 1], 'lik', 1);

% Optimization of hyperparameters
hyp2 = minimize(hyp, @gp, -50, @infGaussLik, meanfunc, covfunc, likfunc,pt, lat);

% Regression using GPR
% yfit is the predicted mean, and ys is the predicted variance
[yfit ys] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc,ps, las, pt);

figure()
plot(ys);
ylabel('variance in prediction')

% Visualization of prediction results
% First output
plotResult(lat(:,1), yfit(:,1))
% Second output
plotResult(lat(:,2), yfit(:,2))
% Third output
plotResult(lat(:,3), yfit(:,3))
% Fourth output
plotResult(lat(:,4), yfit(:,4))
