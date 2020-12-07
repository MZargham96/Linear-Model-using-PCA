clc; close all;

%this code applies softmax and cost-entropy cost function over the pressure data 
%Pressure data has been standard normalized and its dimensions have been
%reduced using Linear PCA

%Pressure Data imported from excel
T1 = readtable('pressure_data_raw2', 'Range', 'A1:GT154');
T2A = table2array(T1);
%%
%sort data into x and y
x = T2A(:,2:end)';
y = T2A(:,1)';
P = size(x,2);

%sorting data into testing and training sets
one = find(y==1);
minusone = find(y==0);
onebythree = round(2/3*length(one));

% training data set and standard normalization
ytrain = [y(one(1,1:onebythree)) y(minusone(1,1:onebythree))];
xtrain = [x(:,one(1,1:onebythree)) x(:,minusone(1,1:onebythree))];

Means = mean(xtrain,2);
S = std(xtrain,1,2);
Xtrain = (xtrain - Means)./S;

%testing set and standard normalization
ytest = [y(one(1,onebythree+1:end)) y(minusone(1,onebythree+1:end))];
xtest = [x(:,one(1,onebythree+1:end)) x(:,minusone(1,onebythree+1:end))];

Means = mean(xtest,2);
S = std(xtest,1,2);
Xtest = (xtest - Means)./S;

% find PCA
Cov = (1/P)*Xtrain*Xtrain'; %Covariance Matrix
[V,D] = eig(Cov);
[D_sorted,id] = sort(diag(D),'descend'); %sorting eigenvalues
A = D_sorted(D_sorted>0.0005);

% choose spanning set
Ctrain = [];
for i = 1:length(A)
    Ctrain = horzcat(Ctrain, V(:,id(i)));
    i = i + 1;
end

% project data  onto new spanning set
wptrain = linsolve(Ctrain'*Ctrain,Ctrain'*Xtrain);
wptest = linsolve(Ctrain'*Ctrain,Ctrain'*Xtest);
%%
%initialization of weights
w0 = randn(length(A)+1,1);

example = 1; %Select example depending on what case you want to run

switch(example)
    case 1 %Cross entropy Cost function
        
        %training the classifier
        f = @(w)CEC(w,wptrain,ytrain);
        [W,fW] = grad_descent(f,w0,0.01,0.000001); 
        min = W(:,end); % final weights obtained through minimization 
        
        %figure showing cost optimization w.r.t iteration 
        cost = CEC(W,wptrain,ytrain);
        plot(cost)
        xlabel('Iteration'); ylabel('Cost/ Error');
        title('Cost minimimization for Pressure Data')
        
        %Defining Threshold for analysis and accuracy measurement
        Ptrain = size(wptrain,2);
        xbartrain = [ones(1,Ptrain); wptrain];
        S = sigmoid(xbartrain'*W(:,end));
        S(S>0.5) = 1;
        S(S<0.5) = 0;
        
        Ctrain = confusionmat(S,ytrain)
        Acc = 0.5*(Ctrain(1,1)/sum(Ctrain(:,1)))...
            + 0.5*(Ctrain(2,2)/sum(Ctrain(:,2)));
        
        Abalancedtrain = Acc*100
        
        Ptest = size(wptest,2);
        xbartest = [ones(1,Ptest); wptest];
        S = sigmoid(xbartest'*W(:,end));
        S(S>0.5) = 1;
        S(S<0.5) = 0;
        
        Ctest = confusionmat(S,ytest)
        Acc = 0.5*(Ctest(1,1)/sum(Ctest(:,1)))...
            + 0.5*(Ctest(2,2)/sum(Ctest(:,2)));
        
        Abalancedtest = Acc*100
        
    case 2 %softmax Cost function       
        ytrain(ytrain==0)=-1;
        ytest(ytest==0)=-1;
        
        %training the classifier
        f = @(w)softmax(w,wptrain,ytrain);
        [W,fW] = grad_descent(f,w0,0.01,0.00001);
        min = W(:,end); % final weights obtained through minimization 

        %figure showing cost optimization w.r.t iteration 
        cost = softmax(W,wptrain,ytrain);
        plot(cost)
       
        %Defining Threshold for analysis and accuracy measurement
        Ptrain = size(wptrain,2);
        xbartrain = [ones(1,Ptrain); wptrain];
        yp = tanh(xbartrain'*W(:,end));
        yp(yp>0) = 1;
        yp(yp<0) = -1;
        
        Ctrain = confusionmat(yp,ytrain)
        Acc = 0.5*(Ctrain(1,1)/sum(Ctrain(:,1)))...
            + 0.5*(Ctrain(2,2)/sum(Ctrain(:,2)));
        
        Abalanced = Acc*100
        
        Ptest = size(wptest,2);
        xbartest = [ones(1,Ptest); wptest];
        yp = tanh(xbartest'*W(:,end));
        yp(yp>0) = 1;
        yp(yp<0) = -1;
        
        Ctest = confusionmat(yp,ytest)
        Acc = 0.5*(Ctest(1,1)/sum(Ctest(:,1)))...
            + 0.5*(Ctest(2,2)/sum(Ctest(:,2)));
        
        Abalanced = Acc*100
end
%%
function cost = softmax(w,wptrain,ytrain)

p = length(ytrain);
xbar = [ones(1,p); wptrain];

cost = (1/p) * ...
    sum( ...
    log(1+exp(-repmat(ytrain',1,size(w,2)).*(xbar'*w))));

end

function cost = CEC(w,wptrain,ytrain)

p = length(ytrain);
xbar = [ones(1,p); wptrain];
sig = sigmoid(xbar'*w);

cost = -1/p * sum( repmat(ytrain',1,size(w,2)) .* log(sig)...
    + (1 - repmat(ytrain',1,size(w,2))) .* log(1-sig));
end

function s = sigmoid(wptrain) % Activiation Function

s = 1./(1+exp(-wptrain));
s(s==1) = .9999;
s(s==0) = .0001;
end

function [W,fW] = grad_descent(f,w0,alpha,error)

k = 2;
W(:,k) = w0;
fW(k) = f(w0);

while abs(fW(k)-fW(k-1)) > error
    grad = approx_grad(f,W(:,k),.000001);
    W(:,k+1) = W(:,k) - alpha*(grad')/norm(grad);
    fW(k+1) = f(W(:,k+1));
    k = k+1;
end
end

function grad = approx_grad(f,w0,delta)

N = length(w0);
dw = eye(N)*delta;
grad = (f(w0 + dw) - f(w0))/ dw;
end