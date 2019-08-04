%%%% IDENTIFICATION EXAMPLE 4 OSLA MCNN

%%%%%%%%%%%%% IDENTIFICATION PROCESS
clc;clear;
k=[1:50000];
f=@(x1,x2,x3,x4,x5)((x1*x2*x3*x5*(x3-1)+x4)./(1+x3^2+x2^2));

NNclass=[5 50 1];
in=NNclass(1);n1=NNclass(2);out=NNclass(3);

W1=randn(in+1,n1);          %%%% Weight initialization
W2=zeros(n1+1,out);

c=0;
b=0;
a=0;
yp=[c b a zeros(1,length(k))];                  
yphat=[c b a zeros(1,length(k))];

u=(-1+2*rand(1,length(k)+3));        %%%% Input to the systems

e1=0.001;
e2=0.6;

lambda=1e-04;
P=(1/lambda)*eye(n1+1);

error_train=zeros(1,length(k)+3);
for i=4:length(k)+3
    i
    yp(i)=f(yp(i-1),yp(i-2),yp(i-3),u(i-1),u(i-2));
    %%%% NEURAL NETWORK
    %Forward Pass
    A1=[1 yp(i-1) yp(i-2) yp(i-3) u(i-1) u(i-2)]*W1;
    y1=tanh(A1);
    v1=[1 y1];
    A2=[1 y1]*W2;
    N=A2;
    %Identification model output
    yphat(i)=N;
    e=-(yphat(i)-yp(i));            %%%% error
    
    if(abs(e)<e1)
        continue;
    elseif(abs(e)>e1 && abs(e)<e2)
        P=P - ((P*v1'*v1*P)./(1+v1*P*v1'));
        W2=W2+e*P*v1';
        error_train(i-3)=e;
    else
        W1=[W1 randn(in+1,1)];
        W2=[W2;0];
        
        A1=[1 yp(i-1) yp(i-2) yp(i-3) u(i-1) u(i-2)]*W1;
        y1=tanh(A1);
        v1=[1 y1];
        A2=[1 y1]*W2;
        N=A2;
        P=[P zeros(size(P,1),1)];
        P=[P;zeros(1,size(P,2))];
        P(end,end)=1/lambda;
        P=P - (P*v1'*v1*P)./(1+v1*P*v1');
        W2=W2+e*P*v1';
        error_train(i-3)=e;
    end
end
%%
plot(yp,'k')
hold on
plot(yphat,'k--','LineWidth',0.5)
xlim([1 200])
title('response of actual plant and identification model(training)');
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex4/osla/train1.png')
%%
%plot different time step
figure;
plot(yp,'k')
hold on
plot(yphat,'k--','LineWidth',0.5)
xlim([41000 41200])
title('response of actual plant and identification model (at a different time step)');
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex4/osla/train2.png')
%%
%Performance Measure
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
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex4/osla/error_train.png')
%%
%saving weights and biases
save('WeightsH1_NP90_ex4_osla','W1','W2');
disp('saved');
%%
clear;
%loading weights and biases
load('WeightsH1_NP90_ex4_osla');
disp('loaded');
%%
k=[1:50000];
f=@(x1,x2,x3,x4,x5)((x1*x2*x3*x5*(x3-1)+x4)./(1+x3^2+x2^2));
c=0;
b=0;
a=0;
yp=[c b a zeros(1,length(k))];                  
yphat=[c b a zeros(1,length(k))];
u1=sin((2*pi).*[1:500+3]./250);
u2=0.8*(sin((2*pi).*[500+4:length(k)+3]./250))+0.2*(sin((2*pi).*[500+4:length(k)+3]./25));
u=[u1 u2];
error_test=zeros(1,length(k)+3);
for i=4:length(k)+3              %%%% This is to show the output like in NP90
    yp(i)=f(yp(i-1),yp(i-2),yp(i-3),u(i-1),u(i-2));
    
    A1=[1 yphat(i-1) yphat(i-2) yphat(i-3) u(i-1) u(i-2)]*W1;
    y1=tanh(A1);
    A2=[1 y1]*W2;
    N=A2;
    yphat(i)=N;
    e=-(yphat(i)-yp(i));
    error_test(i-3)=e;
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
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex4/osla/error_test.png')
%%
figure;
plot(yp,'LineWidth',1.5)
hold on
plot(yphat,':','LineWidth',2.5)
xlim([1 1000])
ylim([-1 1])
title('For given test reference input')
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex4/osla/test.png')