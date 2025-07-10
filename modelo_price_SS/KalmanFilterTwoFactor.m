function [LogLScore] = KalmanFilterTwoFactor(param, InitStateVect, InitStateCov, futuredata, matur, dt)
% Kalman Filter for two factor price process
% the function has a single output (logl) for minimization purposes only,
% filtered variables are obtained via global variables

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization of Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kappa = param(1);
sigmachi = param(2);
riskchi = param(3);
muxi = param(4);
sigmaxi = param(5);
riskxi = param(6);
rho = param(7);
s = param(8:end); %errors, s1, s2, s3, s4, .... in Schwartz and Smith

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define Variables in Transition Equation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Notation: x(t) = c + G*x(t-1) + wt
% wt = N(0,Q)
% Q is a process noise

Ci = [0, muxi*dt]';
Gi = [exp(-kappa*dt), 0; 0, 1];
xx = (1-exp(-2*kappa*dt)) * (sigmachi^2/(2*kappa));
xy = (1-exp(-kappa*dt)) * rho * sigmachi * sigmaxi / kappa;
yy = sigmaxi^2*dt;

Q = [xx, xy; xy, yy];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define Variables in Measurement Equation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Notation: y(t) = d + F*x(t) + vt
% yt = ln(Ft1). ln(Ft2), ..., ln(Ftn)
% d = A(T1), A(T2), A(T3), ..., A(Tn)
% vt = N(0,V) 
% V = sigmaf^2 * I  definition from Andresen and Sollie (2013)
% R = measurement noise; assumed to be 0

FirstAT = (muxi - riskxi) * matur;
SecondAT = -(1-exp(-kappa*matur)) * (riskchi/kappa);
ThirdAT = 0.5*((1-exp(-2*kappa*matur))*(sigmachi^2/kappa) + sigmaxi^2*matur + 2*(1-exp(-kappa*matur))*rho*sigmaxi*sigmachi/kappa);
di = (FirstAT + SecondAT + ThirdAT)';
Fi = [exp(-kappa*matur)', ones(size(matur,2),1)];
R = eye(size(Q,1));
V = diag(s); %therefore measurement error = initial S

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run Kalman Filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%PlaceHolder for Global Variable (to avoid run the KF twice)
global vtgivtvec
global Xtgivtvec

%Initialization
Xtgivt = InitStateVect';
Ptgivt = InitStateCov;
Result_SecondTermLogL = zeros(size(futuredata,1),1);
Result_FirstTermLogL = zeros(size(futuredata,1),1);

vtgivtvec = zeros(size(futuredata));
Xtgivtvec = zeros(size(futuredata,1),2);

for i = 1:size(futuredata,1)
    %Prediction Steps
    Xtgivtmin1 = Ci+Gi * Xtgivt; % 2 x 1 matrix
    Ptgivtmin1 = Gi * Ptgivt * Gi' + R * Q * R'; % 2 x 2 matrix

    %Measurement Steps (Updating Steps)
    yt = futuredata(i,:)'; % Tx1 matrix

    vtgivtmin1 = yt - (di+Fi*Xtgivtmin1); %difference between data and prediction
    
    Htgivtmin1 = Fi*Ptgivtmin1*Fi' + V; %transition matrix
    Kt = Ptgivtmin1 * Fi' * inv(Htgivtmin1); %Kalman Gain
    
    Xtgivt = Xtgivtmin1 + Kt*vtgivtmin1; %update the state variables
    Ptgivt = Ptgivtmin1 - Kt*Fi*Ptgivtmin1; %update the covariance
    
    Result_SecondTermLogL(i,:) = vtgivtmin1' * inv(Htgivtmin1) * vtgivtmin1; %save result for second term LogL
    Result_FirstTermLogL(i,:) = det(Htgivtmin1); % save result for first term logL
    
    vtgivt = yt - (di+Fi*Xtgivt);
    vtgivtvec(i,:) = vtgivt ;
    Xtgivtvec(i,:) = Xtgivt;
    
end

LogLScore = -0.5*sum(log(2*pi*Result_FirstTermLogL)) - 0.5*sum(Result_SecondTermLogL);
LogLScore = -LogLScore;
end



