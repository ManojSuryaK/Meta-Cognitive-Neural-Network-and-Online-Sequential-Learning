%%%% IDENTIFICATION EXAMPLE 2 PART B OSLA

%%%%%%%%%%%%% IDENTIFICATION PROCESS
clc;clear;
k=[1:50000];
f=@(x,y)(x*y*(x+2.5)/(1+x^2+y^2));
difftanh=@(x)(sech(x));

NNclass=[2 150 1];
in=NNclass(1);n1=NNclass(2);out=NNclass(3);

W1=randn(in+1,n1);          %%%% Weight initialization
W2=zeros(n1+1,out);

b=normrnd(0,2,1,1);
a=normrnd(0,2,1,1);
yp=[b a zeros(1,length(k))];                  
yphat=[b a zeros(1,length(k))];
u=(-2+4*rand(1,length(k)+2));        %%%% Input to the systems

lambda=1e-07;
P=(1/lambda)*eye(n1+1);

error_train=zeros(1,length(k)+2);
for i=3:length(k)+2
    yp(i)=f(yp(i-1),yp(i-2))+u(i-1);
    %Forward Pass
    A1=[1 yp(i-1) yp(i-2)]*W1;
    y1=tanh(A1);
    v1=[1 y1];
    A2=[1 y1]*W2;
    N=A2;
    %Identification model output
    yphat(i)=N+u(i-1);
    e=-(yphat(i)-yp(i));            %%%% error
    P=P - ((P*v1'*v1*P)./(1+v1*P*v1'));
    W2=W2+e*P*v1';
    error_train(i-2)=e;
end
plot(yp)
hold on
plot(yphat)
xlim([1 200])
title('response of actual plant and identification model(training)');
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex2/osla/train1.png')
%%
%plot different time step
figure;
plot(yp)
hold on;
plot(yphat)
xlim([41000 41200])
title('response of actual plant and identification model (at a different time step)');
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex2/osla/train2.png')
%%
%Performance Measure
figure;
plot(error_train)
title('Training error')
rmse1=rms(error_train(1:10000));
rmse2=rms(error_train(10001:end));
fprintf('Training RMS error 1 = %f\n',rmse1);
fprintf('Training RMS error 2 = %f\n\n',rmse2);
disp('Variance Accounted For (training)');
VAF=(1-var(yp-yphat)/var(yp))*100;
disp(VAF);
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex2/osla/error_train.png')
%%
%saving weights and biases
save('WeightsH1_NP90_ex2b_osla','W1','W2');
disp('saved');
%%
clear;
%loading weights and biases
load('WeightsH1_NP90_ex2b_osla');
disp('loaded');
%%
k=[1:10000];
f=@(x,y)(x*y*(x+2.5)/(1+x^2+y^2));
difftanh=@(x)(sech(x));

b=normrnd(0,2,1,1);
a=normrnd(0,2,1,1);
yp=[b a zeros(1,length(k))];                  
yphat=[b a zeros(1,length(k))];
error_test=zeros(1,length(k));
for i=3:length(k)+2
    u=sin(2*pi*(i-3)/250);              %%%% This is to show the output like in NP90
    if(i>250)
        u=0.5*(u+sin(2*pi*(i-3)/25));
    end
    yp(i)=f(yp(i-1),yp(i-2))+u;
    
    A1=[1 yphat(i-1) yphat(i-2)]*W1;
    y1=tanh(A1);
    A2=[1 y1]*W2;
    N=A2;
    
    yphat(i)=N+u;
    e=-(yphat(i)-yp(i));
    error_test(i-2)=e;
end
%%
%Performance Measures
figure;
plot(error_test)
title('Testing error');
rmse1=rms(error_test(1:1000));
rmse2=rms(error_test(1001:end));
fprintf('Testing RMS error 1 = %f\n',rmse1);
fprintf('Testing RMS error 2 = %f\n\n',rmse2);
disp('Variance Accounted For (testing)');
VAF=(1-var(yp-yphat)/var(yp))*100;
disp(VAF);
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex2/osla/error_test.png')
%%
figure;
plot(yp);
hold on;
plot(yphat);
xlim([1 1000])
title('response of actual plant and identified model to a different input(testing)')
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex2/osla/test.png')