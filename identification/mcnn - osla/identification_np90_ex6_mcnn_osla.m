clc;clear;
k=1:50000;

yp=[0 0 zeros(1,length(k))];
yphat=[0 0 zeros(1,length(k))];

NNclass=[1 20 1];
in=NNclass(1);n1=NNclass(2);out=NNclass(3);

W1=randn(in+1,n1);
W2=zeros(n1+1,out);

lambda=2.5e-03;
P=(1/lambda)*eye(n1+1);

e1=0.015;
e2=5;

u=-1+2*rand(1,length(k)+2);

f=@(u)((u-0.8)*u*(u+0.5));
error_train=zeros(1,length(k));

for i=3:length(k)+2
    i
    yp(i)=0.8*yp(i-1)+f(u(i-1));
    
    A1=[1 u(i-1)]*W1;
    y1=tanh(A1);
    v1=[1 y1];
    A2=v1*W2;
    N=A2;
    
    yphat(i)=0.8*yphat(i-1)+N;
    
    e=-(yphat(i)-yp(i));
    
    if(abs(e)<e1)
        continue;
    elseif(abs(e)>e1 && abs(e)<e2)
        P=P - (P*v1'*v1*P)./(1+v1*P*v1');
        W2=W2+e*P*v1';
        error_train(i-2)=e;
    else
        W1=[W1 randn(2,1)];
        W2=[W2;0];
        
        A1=[1 u(i-1)]*W1;
        y1=tanh(A1);
        v1=[1 y1];
        A2=v1*W2;
        N=A2;
        
        P=[P zeros(size(P,1),1)];
        P=[P;zeros(1,size(P,2))];
        P(end,end)=1/lambda;
        P=P - (P*v1'*v1*P)./(1+v1*P*v1');
        W2=W2+e*P*v1';
        error_train(i-2)=e;
    end
end
%%
plot(yp,'k')
hold on
plot(yphat,'k--','LineWidth',0.5)
xlim([1 250])
title('response of actual plant and identification model(training)');
legend('actual plant output','identification model output');
%%
figure;
plot(yp,'k')
hold on
plot(yphat,'k--','LineWidth',0.5)
xlim([41000 41200])
title('response of actual plant and identification model (at a different time step)');
legend('actual plant output','identification model output');
%%
%Performance Measures (Training)
figure;
plot(error_train,'k')
xlim([1 500])
title('Training error')
rmse1=rms(error_train(1:10000));
rmse2=rms(error_train(10001:end));
fprintf('Training RMS error 1 = %f\n',rmse1);
fprintf('Training RMS error 2 = %f\n\n',rmse2);
disp('Variance Accounted For (training)');
VAF=(1-var(yp-yphat)/var(yp))*100;
disp(VAF);
%%
%saving weights
save('WeightsH2_NP90_ex6_mcnn','W1','W2');
disp('saved');
%%
%loading weights
load('WeightsH2_NP90_ex6_mcnn');
disp('loaded');
%%
%response of identification model(hopefully identified system) to different input
k=[1:5000];
f=@(u)((u-0.8)*u*(u+0.5));
yp=[0 0 zeros(1,length(k))];
yphat=[0 0 zeros(1,length(k))];
u=@(k)(sin((2*pi).*k./25));
error_test=zeros(1,length(k));
for i=3:length(k)+2
    yp(i)=0.8*yp(i-1)+f(u(i-1)); %%%% Plant output
    %Forward Pass
    A1=[1 u(i-1)]*W1;
    y1=tanh(A1);
    A2=[1 y1]*W2;
    N=A2;
    yphat(i)=0.8*yphat(i-1)+N;   %%%% INDEPENDENT identfication model output
    e=-(yphat(i)-yp(i));
    error_test(i-2)=e;
end
%%
%Performance Measures
figure;
plot(error_test,'k')
title('Testing error');
rmse1=rms(error_test(1:1000));
rmse2=rms(error_test(1001:end));
fprintf('Testing RMS error 1 = %f\n',rmse1);
fprintf('Testing RMS error 2 = %f\n\n',rmse2);
disp('Variance Accounted For (testing)');
VAF=(1-var(yp-yphat)/var(yp))*100;
disp(VAF);
%%
figure;
plot(yp,'LineWidth',1.5)
hold on
plot(yphat,'--','LineWidth',2.5)
xlim([1 100])
title('For given test reference input')
legend('actual plant output','identification model output');