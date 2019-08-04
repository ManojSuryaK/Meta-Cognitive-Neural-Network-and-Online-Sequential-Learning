clc;clear;

k=[1:50000];
f=@(x,y)(x*y*(x+2.5)/(1+x^2+y^2));
difftanh=@(x)(sech(x).^2);

NNclass=[2 20 10 1];
in=NNclass(1);n1=NNclass(2);n2=NNclass(3);out=NNclass(4);

W1=randn(in+1,n1);          %%%% Weight initialization
W2=randn(n1+1,n2);
W3=zeros(n2+1,out);

eta=0.1;                           %%%%%%%%%% HOLY SHIT

yp=[0 0 zeros(1,length(k))];                  
yphat=[0 0 zeros(1,length(k))];

u=(-2+4*rand(1,length(k)+2));        %%%% Input to the systems
error_train=zeros(1,length(k));
for i=3:length(k)+2
    yp(i)=f(yp(i-1),yp(i-2))+u(i-1);
    %%%% NEURAL NETWORK
    %Forward Pass
    A1=[1 yp(i-1) yp(i-2)]*W1;
    y1=tanh(A1);
    A2=[1 y1]*W2;
    y2=tanh(A2);
    A3=[1 y2]*W3;
    N=A3;
    %Identification model output
    yphat(i)=N+u(i-1);
    e=-(yphat(i)-yp(i));            %%%% error
    %Backward Pass
    del3=e;
    del2=difftanh(A2).*(del3*W3(2:end,:)');
    del1=difftanh(A1).*(del2*W2(2:end,:)');
    Jw3=[1 y2]'*del3;
    Jw2=[1 y1]'*del2;
    Jw1=[1 yp(i-1) yp(i-2)]'*del1;
    %Weight Update
    W3=W3+eta*Jw3;
    W2=W2+eta*Jw2;
    W1=W1+eta*Jw1;
    error_train(i-2)=e;
end
plot(yp(1:200))
hold on
plot(yphat(1:200))
title('response of actual plant and identification model(training)');
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex2/bpa2/train1.png')
%%
%plot different time step
figure;
plot(yp(41000:41200))
hold on;
plot(yphat(41000:41200))
title('response of actual plant and identification model (at a different time step)');
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex2/bpa2/train2.png')
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
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex2/bpa2/error_train.png')
%%
%saving weights and biases
save('WeightsH2_NP90_ex2b','W1','W2','W3');
disp('saved');
%%
clear;
%loading weights and biases
load('WeightsH2_NP90_ex2b');
disp('loaded');
%%
k=[1:10000];
f=@(x,y)(x*y*(x+2.5)/(1+x^2+y^2));

yp=[0 0 zeros(1,length(k))];                  
yphat=[0 0 zeros(1,length(k))];
error_test=zeros(1,length(k));
%u=@(t)(sin(2*pi*t/250)+(t>=250).*(sin(2*pi*t/25)));
u=@(t)(sin(2*pi*t/25));
for i=3:length(k)+2
    yp(i)=f(yp(i-1),yp(i-2))+u(i-3);
    
    A1=[1 yphat(i-1) yphat(i-2)]*W1;
    y1=tanh(A1);
    A2=[1 y1]*W2;
    y2=tanh(A2);
    A3=[1 y2]*W3;
    N=A3;
    
    yphat(i)=N+u(i-3);
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
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex2/bpa2/error_test.png')
%%
figure;
plot(yp(1:100));
hold on;
plot(yphat(1:100));
title('response of actual plant and identified model to a different input(testing)')
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex2/bpa2/test.png')