%%%% IDENTIFICATION EXAMPLE 4 ONE LAYER

%%%%%%%%%%%%% IDENTIFICATION PROCESS
clc;clear;
k=[1:50000];
f=@(x1,x2,x3,x4,x5)((x1*x2*x3*x5*(x3-1)+x4)./(1+x3^2+x2^2));
difftanh=@(x)(sech(x));

NNclass=[5 20 1];
in=NNclass(1);n1=NNclass(2);out=NNclass(3);

W1=randn(in+1,n1);          %%%% Weight initialization
W2=zeros(n1+1,out);

eta=0.1;                           %%%%%%%%%% HOLY SHIT
c=0;
b=0;
a=0;
yp=[c b a zeros(1,length(k))];                  
yphat=[c b a zeros(1,length(k))];

u=(-1+2*rand(1,length(k)+3));        %%%% Input to the systems
error_train=zeros(1,length(k)+3);
for i=4:length(k)+3
    yp(i)=f(yp(i-1),yp(i-2),yp(i-3),u(i-1),u(i-2));
    %%%% NEURAL NETWORK
    %Forward Pass
    A1=[1 yp(i-1) yp(i-2) yp(i-3) u(i-1) u(i-2)]*W1;
    y1=tanh(A1);
    A2=[1 y1]*W2;
    N=A2;
    %Identification model output
    yphat(i)=N;
    e=-(yphat(i)-yp(i));            %%%% error
    %Backward Pass
    del2=e;
    del1=difftanh(A1).*(del2*W2(2:end,:)');
    Jw2=[1 y1]'*del2;               %%%% These are not gradients, these are cray-dients
    Jw1=[1 yp(i-1) yp(i-1) yp(i-3) u(i-1) u(i-2)]'*del1;
    %Weight Update
    W2=W2+eta*Jw2;
    W1=W1+eta*Jw1;
    error_train(i-3)=e;
end
plot(yp)
hold on
plot(yphat)
xlim([1 200])
title('response of actual plant and identification model(training)');
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex4/bpa1/train1.png')
%%
%plot different time step
figure;
plot(yp)
hold on;
plot(yphat)
xlim([41000 41200])
title('response of actual plant and identification model (at a different time step)');
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex4/bpa1/train2.png')
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
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex4/bpa1/error_train.png')
%%
%saving weights and biases
save('WeightsH1_NP90_ex4_one_layer','W1','W2');
disp('saved');
%%
clear;
%loading weights and biases
load('WeightsH1_NP90_ex4_one_layer');
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
plot(error_test)
title('Testing error');
rmse1=rms(error_test(1:1000));
rmse2=rms(error_test(1001:end));
fprintf('Testing RMS error 1 = %f\n',rmse1);
fprintf('Testing RMS error 2 = %f\n\n',rmse2);
disp('Variance Accounted For (testing)');
VAF=(1-var(yp-yphat)/var(yp))*100;
disp(VAF);
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex4/bpa1/error_test.png')
%%
figure;
plot(yp);
hold on;
plot(yphat);
xlim([1 1000])
title('response of actual plant and identified model to a different input(testing)')
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex4/bpa1/test.png')