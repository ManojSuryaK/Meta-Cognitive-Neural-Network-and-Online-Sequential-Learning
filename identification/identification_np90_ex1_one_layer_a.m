%%%% IDENTIFICATION EXAMPLE 1 ONE LAYER
clc;clear;
%%%%%%%%%%%%% IDENTIFICATION PROCESS
k=[1:100000];

a=0;b=0;
yp=[b a zeros(1,length(k))];        %%%% plant trajectory 
yphat=[b a zeros(1,length(k))];     %%%% identification model trajectory

NNclass=[1 5 1];
in=NNclass(1);n1=NNclass(2);out=NNclass(3);

W1=randn(in+1,n1);          %%%% Weight initialization
W2=zeros(n1+1,out);

eta=0.1;                           %%%%%%%%%% HOLY SHIT

u=@(k)(-1+2*rand);       %%%% Training Input to the systems
%u=@(k)(sin(2*pi*k/250));
%f=@(u)(u^3+0.3*u^2-0.4*u);          %%%% function to be approximated
f=@(u)(0.6*sin(pi*u)+0.3*sin(3*pi*u)+0.1*sin(5*pi*u));
error_train=zeros(1,length(k));
difftanh=@(x)(sech(x));
for i=3:length(k)+2
    yp(i)=0.3*yp(i-1)+0.6*yp(i-2)+f(u(i-3));
    %%%% NEURAL NETWORK(i suck at this shit)
    %Forward Pass
    A1=[1 u(i-1)]*W1;
    y1=tanh(A1);
    A2=[1 y1]*W2;
    N=A2;
    %Identification model output
    yphat(i)=0.3*yp(i-1)+0.6*yp(i-2)+N;
    e=-(yphat(i)-yp(i));            %%%% error
    %Backward Pass
    if(i<=k(end))
        del2=e;
        del1=difftanh(A1).*(del2*W2(2:end,:)');
        Jw2=[1 y1]'*del2;               %%%% These are not gradients, these are cray-dients
        Jw1=[1 u(i-3)]'*del1;
        %Weight Update
        W2=W2+eta*Jw2;
        W1=W1+eta*Jw1;
    end
    error_train(i-2)=e;
end
plot(yp)
hold on
plot(yphat)
xlim([1 500])
title('response of actual plant and identification model(training)');
legend('actual plant output','identification model output');
%%
%plot different time step
figure;
plot(yp)
hold on;
plot(yphat)
xlim([41000 41500])
title('response of actual plant and identification model (at a different time step)');
legend('actual plant output','identification model output');
%%
%Performance Measure - RMSE and VAF
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
%%
%saving weights
save('WeightsH1_NP90_ex1_one_layer','W1','W2');
disp('saved');
%%
%loading weights
load('WeightsH1_NP90_ex1_one_layer');
disp('loaded');
%%
%response of identification model(hopefully identified system) to different input
k=[1:20000];
%f=@(u)(u^3+0.3*u^2-0.4*u);
f=@(u)(0.6*sin(pi*u)+0.3*sin(3*pi*u)+0.1*sin(5*pi*u));
u=@(k)sin(2*pi*k/250);
b=0; a=0;
yp=[b a zeros(1,length(k))];
yphat=[b a zeros(1,length(k))];
error_test=zeros(1,length(k));
for i=3:length(k)+2
    yp(i)=0.3*yp(i-1)+0.6*yp(i-2)+f(u(i-3)); %%%% Plant output
    %Forward Pass
    A1=[1 u(i-3)]*W1;
    y1=tanh(A1);
    A2=[1 y1]*W2;
    N=A2;
    yphat(i)=0.3*yphat(i-1)+0.6*yphat(i-2)+N;   %%%% INDEPENDENT identfication model output
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
%%
figure;
plot(yp);
hold on;
plot(yphat);
xlim([1 1000])
title('response of actual plant and identified model to a different input(testing)')
legend('actual plant output','identification model output');