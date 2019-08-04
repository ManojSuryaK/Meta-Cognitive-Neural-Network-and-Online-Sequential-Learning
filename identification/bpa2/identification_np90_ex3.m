%%%% IDENTIFICATION EXAMPLE 3

%%%%%%%%%%%%% IDENTIFICATION PROCESS

clc;clear;
k=1:50000;
f=@(x)(x./(1+x.^2));
g=@(x)(x.^3);
difftanh=@(x)(sech(x).^2);

NN_1class=[1 20 10 1];
NN_2class=[1 20 10 1];
in_1=NN_1class(1);n1_1=NN_1class(2);n2_1=NN_1class(3);out_1=NN_1class(4);
in_2=NN_1class(1);n1_2=NN_2class(2);n2_2=NN_2class(3);out_2=NN_2class(4);

W1_1=randn(in_1+1,n1_1);
W2_1=randn(n1_1+1,n2_1);
W3_1=randn(n2_1+1,out_1);

W1_2=randn(in_2+1,n1_2);
W2_2=randn(n1_2+1,n2_2);
W3_2=randn(n2_2+1,out_2);

yp=[0 zeros(1,length(k))];
yphat=[0 zeros(1,length(k))];

u=[0 -2+4*rand(1,length(k))];

eta=0.01;

error_train=[];

for i=2:length(k)+1
    fx=f(yp(i-1));
    gx=g(u(i-1));
    yp(i)=fx+gx;
    
    A1_1=[1 yp(i-1)]*W1_1;
    y1_1=tanh(A1_1);
    A2_1=[1 y1_1]*W2_1;
    y2_1=tanh(A2_1);
    A3_1=[1 y2_1]*W3_1;
    N_1=A3_1;
    
    A1_2=[1 u(i-1)]*W1_2;
    y1_2=tanh(A1_2);
    A2_2=[1 y1_2]*W2_2;
    y2_2=tanh(A2_2);
    A3_2=[1 y2_2]*W3_2;
    N_2=A3_2;
    
    yphat(i)=N_1+N_2;

    e=-(yphat(i)-yp(i));
    
    del3_1=e;
    del2_1=difftanh(A2_1).*(del3_1*W3_1(2:end,:)');
    del1_1=difftanh(A1_1).*(del2_1*W2_1(2:end,:)');
    Jw3_1=[1 y2_1]'*del3_1;
    Jw2_1=[1 y1_1]'*del2_1;
    Jw1_1=[1 yp(i-1)]'*del1_1;
    W3_1=W3_1+eta*Jw3_1;
    W2_1=W2_1+eta*Jw2_1;
    W1_1=W1_1+eta*Jw1_1;
    
    del3_2=e;
    del2_2=difftanh(A2_2).*(del3_2*W3_2(2:end,:)');
    del1_2=difftanh(A1_2).*(del2_2*W2_2(2:end,:)');
    Jw3_2=[1 y2_2]'*del3_2;
    Jw2_2=[1 y1_2]'*del2_2;
    Jw1_2=[1 u(i-1)]'*del1_2;
    W3_2=W3_2+eta*Jw3_2;
    W2_2=W2_2+eta*Jw2_2;
    W1_2=W1_2+eta*Jw1_2;
    
    error_train(i-1)=e;
end
plot(yp)
hold on
plot(yphat)
xlim([1 500])
title('response of actual plant and identification model(training)');
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex3/bpa2/train1.png')
%%
%plot different time step
figure;
plot(yp)
hold on;
plot(yphat)
xlim([41000 41500])
title('response of actual plant and identification model (at a different time step)');
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex3/bpa2/train2.png')
%%
%Performance Measures (Training)
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
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex3/bpa2/error_train.png')
%%
%saving weights and biases
save('WeightsH2_NP90_ex3','W1_1','W2_1','W3_1','W1_2','W2_2','W3_2');
disp('saved');
%%
clear;
%loading weights and biases
load('WeightsH2_NP90_ex3');
disp('loaded');
%%
k=1:10000;
f=@(x)(x./(1+x.^2));
g=@(x)(x.^3);

yp=[0 zeros(1,length(k))];
yphat=[0 zeros(1,length(k))];

u=[0 sin((2*pi).*k./25)+sin((2*pi).*k./10)];

error_test=[];
for i=2:length(k)+1
    fx=f(yp(i-1));
    gx=g(u(i-1));
    yp(i)=fx+gx;
    
    A1_1=[1 yphat(i-1)]*W1_1;
    y1_1=tanh(A1_1);
    A2_1=[1 y1_1]*W2_1;
    y2_1=tanh(A2_1);
    A3_1=[1 y2_1]*W3_1;
    N_1=A3_1;
    
    A1_2=[1 u(i-1)]*W1_2;
    y1_2=tanh(A1_2);
    A2_2=[1 y1_2]*W2_2;
    y2_2=tanh(A2_2);
    A3_2=[1 y2_2]*W3_2;
    N_2=A3_2;
    
    yphat(i)=N_1+N_2;
    
    e=-(yphat(i)-yp(i));
    error_test(i-1)=e;
end
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
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex3/bpa2/error_test.png')
%%
figure;
plot(yp);
hold on;
plot(yphat);
xlim([1 100]);
title('response of actual plant and identified model to a different input(testing)')
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex3/bpa2/test.png')
%%
figure;
k=[-2:0.1:2];
g=@(u)(u^3);
gx=[zeros(1,length(k))];
gxhat=[zeros(1,length(k))];
for i=1:length(k)
    gx(i)=g(k(i));
    
    A1_2=[1 k(i)]*W1_2;
    y1_2=tanh(A1_2);
    A2_2=[1 y1_2]*W2_2;
    y2_2=tanh(A2_2);
    A3_2=[1 y2_2]*W3_2;
    gxhat(i)=A3_2;
end
subplot(121);
plot(k,gx)
hold on
plot(k,gxhat)
title('Plot of g and ghat');
legend('g','ghat');
%%
k=[-10:0.1:10];
f=@(x)(x/(1+x^2));
fx=[zeros(1,length(k))];
fxhat=[zeros(1,length(k))];
for i=1:length(k)
    fx(i)=f(k(i));

    A1_1=[1 k(i)]*W1_1;
    y1_1=tanh(A1_1);
    A2_1=[1 y1_1]*W2_1;
    y2_1=tanh(A2_1);
    A3_1=[1 y2_1]*W3_1;
    fxhat(i)=A3_1;
end
subplot(122);
plot(k,fx)
hold on
plot(k,fxhat)
title('Plot of f and fhat');
legend('f','fhat');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex3/bpa2/gf and hats.png');