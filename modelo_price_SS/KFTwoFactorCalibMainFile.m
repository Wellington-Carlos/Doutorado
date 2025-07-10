%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This Matlab Script estimates the parameters of the model presented in Schwartz-Smith 
% (2000) - paper(Short-Term Variations and Long-Term Dynamics in Commodity Prices).
% The estimation of the parameters is done using Kalman Filter.
% NOTE: it can take sometimes to complete the calculation (depend on
% the calibration data that you use)
% 
% Produced by Philip Thomas (First Prod: May 2014)
% 
% Contact: philip.thomas@DecisionAnalyst.me
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialization
clc;
clear;
format short;

% define global variable from KalmanFilterTwoFactor Function
global vtgivtvec
global Xtgivtvec

%input data
Data = csvread('WTIJan1990Sep2016.csv');
deltat = 1/52;
matur = [1/12, 3/12, 5/12, 9/12, 13/12, 17/12];

%initialize parameters

kappa = 1;
sigmachi = 0.01;
riskchi = 0.01;
muxi = 0.01;
sigmaxi = 0.01;
riskxi = 0.01;
rho = 0.1;
s = 0.001;

param_initial = [kappa, sigmachi, riskchi, muxi, sigmaxi, riskxi, rho, repmat(s,1,size(Data,2))];
InitStateVect = [0.28, 3.5]; %[chi, xi]
InitStateCov = repmat(0.01, 2, 2);

%%%setting up optimization%%%%%%%%%%

%setting up boundary

%[kappa, sigmachi, riskchi, muxi, sigmaxi, riskxi, rho, s1, s2, s3, ...., sn]
lowerbound = zeros(1, 7+size(Data,2));
lowerbound(1,1:7) = [0, 0, -Inf, -Inf, 0, -Inf, -1];
lowerbound(1,8:end) = 0.00000001;

upperbound = zeros(1,7+size(Data,2));
upperbound(1,1:7) = [Inf, Inf, Inf, Inf, Inf, Inf, 1];
upperbound(1,8:end) = Inf;

%Running MLE

options = optimset('Algorithm', 'interior-point','Display', 'off');
LogLKalman = @(param) KalmanFilterTwoFactor(param, InitStateVect, InitStateCov, Data, matur, deltat); 
[param_optimized,logL] = fmincon(LogLKalman, param_initial,[],[],[],[],lowerbound,upperbound,[], options);

vec = Xtgivtvec;
%processing result
MSE = mean(mean(vtgivtvec));
%hessnum = hessianfunc(LogLKalman, param_optimized);
%StdError = sqrt(diag(inv(hessnum)));
%StdError = real(StdError);

%print result
kappa = param_optimized(1);
sigmachi = param_optimized(2);
riskchi = param_optimized(3);
muxi = param_optimized(4);
sigmaxi = param_optimized(5);
riskxi = param_optimized(6);
rho = param_optimized(7);
chi0 = Xtgivtvec(size(Xtgivtvec,1),1);
xi0 = Xtgivtvec(size(Xtgivtvec,1),2);

Calbration_param = [kappa;sigmachi;chi0;riskchi;muxi;sigmaxi;xi0;riskxi;rho];






