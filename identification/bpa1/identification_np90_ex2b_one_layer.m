clc;clear;

k=[1:50000];
f=@(x,y)(x*y*(x+2.5)/(1+x^2+y^2));
difftanh=@(x)(sech(x).^2);

NNclass=[2 50 1];
in=NNclass(1);n1=NNclass(2);out=NNclass(3);

W1=randn(in+1,n1);
W2=zeros(n1+1,out);

eta=0.01;          

yp=[0 0 zeros(1,length(k))];                  
yphat=[0 0 zeros(1,length(k))];

u=(-2+4*rand(1,length(k)+2));
error_train=zeros(1,length(k));
for i=3:length(k)+2
    yp(i)=f(yp(i-1),yp(i-2))+u(i-1);
   
    A1=[1 yp(i-1) yp(i-2)]*W1;
    y1=tanh(A1);
    A2=[1 y1]*W2;
    N=A2;
    
    yphat(i)=N+u(i-1);
    e=-(yphat(i)-yp(i));
    
    del2=e;
    del1=difftanh(A1).*(del2*W2(2:end,:)');
    Jw2=[1 y1]'*del2; 
    Jw1=[1 yp(i-1) yp(i-2)]'*del1;
    
    W2=W2+eta*Jw2;
    W1=W1+eta*Jw1;
    error_train(i-2)=e;
end
%%{
plot(yp)
hold on
plot(yphat)
xlim([1 200])
title('response of actual plant and identification model(training)');
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex2/bpa1/train1.png')
%%
%plot different time step
figure;
plot(yp)
hold on;
plot(yphat)
xlim([41000 41200])
title('response of actual plant and identification model (at a different time step)');
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex2/bpa1/train2.png')
%%
%Performance Measure
figure;
plot(error_train)
title('Training error')
%}
rmse1=rms(error_train(1:10000));
rmse2=rms(error_train(10001:end));
%rmstrain1(q)=rmse1;
%rmstrain2(q)=rmse2;
%%{
fprintf('Training RMS error 1 = %f\n',rmse1);
fprintf('Training RMS error 2 = %f\n\n',rmse2);
disp('Variance Accounted For (training)');
%}
VAF=(1-var(yp-yphat)/var(yp))*100;
%VAFtrain(q)=VAF;
%%{
disp(VAF);
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex2/bpa1/error_train.png')
%%
%saving weights and biases
save('WeightsH1_NP90_ex2b_one_layer','W1','W2');
disp('saved');
%%
clear;
%loading weights and biases
load('WeightsH1_NP90_ex2b_one_layer');
disp('loaded');
%%
%}
k=1:50000;
f=@(x,y)(x*y*(x+2.5)/(1+x^2+y^2));
difftanh=@(x)(sech(x));

b=normrnd(0,2,1,1);
a=normrnd(0,2,1,1);
yp=[b a zeros(1,length(k))];                  
yphat=[b a zeros(1,length(k))];
error_test=zeros(1,length(k));
for i=3:length(k)+2
    u=sin(2*pi*(i-3)/25);   
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
%%{
figure;
plot(error_test)
title('Testing error');
%}
rmse1=rms(error_test(1:1000));
rmse2=rms(error_test(1001:end));
%rmstest1(q)=rmse1;
%rmstest2(q)=rmse2;
%%{
fprintf('Testing RMS error 1 = %f\n',rmse1);
fprintf('Testing RMS error 2 = %f\n\n',rmse2);
disp('Variance Accounted For (testing)');
%}
VAF=(1-var(yp-yphat)/var(yp))*100;
%VAFtest(q)=VAF;
%%{
disp(VAF);
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex2/bpa1/error_test.png')
%%
figure;
plot(yp);
hold on;
plot(yphat);
xlim([1 100])
title('response of actual plant and identified model to a different input(testing)')
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex2/bpa1/test.png')
%}
%{
end
%%
avgrmstrain1=mean(rmstrain1)
avgrmstrain2=mean(rmstrain2)
avgrmstest1=mean(rmstest1)
avgrmstest2=mean(rmstest2)
avgVAFtrain=mean(VAFtrain)
avgVAFtest=mean(VAFtest)
%}