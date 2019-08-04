%%%% IDENTIFICATION EXAMPLE 1 OSLA
clc;clear;
%%%%%%%%%%%%% IDENTIFICATION PROCESS
k=[1:50000];

a=0;b=0;
yp=[b a zeros(1,length(k))];        
yphat=[b a zeros(1,length(k))];     

NNclass=[1 20 1];
in=NNclass(1);n1=NNclass(2);out=NNclass(3);

W1=randn(in+1,n1);          
W2=zeros(n1+1,out);

u=(-1+2*rand(1,length(k)+2)); 
%f=@(u)(u^3+0.3*u^2-0.4*u);    
f=@(u)(0.6*sin(pi*u)+0.3*sin(3*pi*u)+0.1*sin(5*pi*u));

e1=0.001;
e2=0.50;

lambda=1e-06;
P=(1/lambda)*eye(n1+1);

for i=3:length(k)+2
    i
    yp(i)=0.3*yp(i-1)+0.6*yp(i-2)+f(u(i-1));

    v0=[1 u(i-1)];
    A1=v0*W1;
    y1=tanh(A1);
    v1=[1 y1];
    A2=v1*W2;
    N=A2;

    yphat(i)=0.3*yp(i-1)+0.6*yp(i-2)+N;
    e=-(yphat(i)-yp(i));           
    
    if(abs(e)<e1)
        continue;
    elseif(abs(e)>e1 && abs(e)<e2)
        P=P - ((P*v1'*v1*P)./(1+v1*P*v1'));
        W2=W2+e*P*v1';
        error_train(i-2)=e;
    else
        W1=[W1 randn(in+1,1)];
        W2=[W2;0];
        
        v0=[1 u(i-1)];
        A1=v0*W1;
        y1=tanh(A1);
        v1=[1 y1];
        A2=v1*W2;
        N=A2;
        P=[P zeros(size(P,1),1)];
        P=[P;zeros(1,size(P,2))];
        P(end,end)=1/lambda;
        P=P - ((P*v1'*v1*P)./(1+v1*P*v1'));
        W2=W2+e*P*v1';
    end
end
plot(yp,'k')
hold on
plot(yphat,'k--','LineWidth',0.5)
xlim([1 100])
title('response of actual plant and identification model(training)');
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex1/osla/train1.png')
%%
%plot different time step
figure;
plot(yp,'k')
hold on
plot(yphat,'k--','LineWidth',0.5)
xlim([41000 41100])
title('response of actual plant and identification model (at a different time step)');
legend('actual plant output','identification model output');
%saveas(gcf,'C:/Users/tarun/Documents/id/figs/ex1/osla/train2.png')
%%
%Performance Measure - RMSE and VAF
figure;
plot(error_train,'k')
xlim([1 100])
title('Training error')
rmse1=rms(error_train(1:10000));
rmse2=rms(error_train(10001:end));

fprintf('Training RMS error 1 = %f\n',rmse1);
fprintf('Training RMS error 2 = %f\n\n',rmse2);
disp('Variance Accounted For (training)');
VAF=(1-var(yp-yphat)/var(yp))*100;
disp(VAF);
%%
figure;
plot(abs(error_train),'k')
xlim([1 100])
%%
%{
absol_e=abs(error_train);
M=max(absol_e)
ind=find(absol_e<0.05);
conv_time=0;
for i=1:length(ind)
    z=diff(ind(i:i+100))==1
    if sum(z)==100
        conv_time=ind(i);
        break;
    else
        x=find(z==0);
        i=x(end)
    end
end
conv_time
%}
%%
%response of identification model(hopefully identified system) to different input
k=[1:20000];
%f=@(u)(u^3+0.3*u^2-0.4*u);
f=@(u)(0.6*sin(pi*u)+0.3*sin(3*pi*u)+0.1*sin(5*pi*u));
yp=[0 0 zeros(1,length(k))];
yphat=[0 0 zeros(1,length(k))];
error_test=zeros(1,length(k));
for i=3:length(k)+2
    u=sin(2*pi*(i-3)/250);              %%%% This is to show the output like in NP90
    %if(i>250)
    %   u=0.5*(u+sin(2*pi*(i-3)/25));
    %end
    yp(i)=0.3*yp(i-1)+0.6*yp(i-2)+f(u); %%%% Plant output
    %Forward Pass
    A1=[1 u]*W1;
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
xlim([1 1000])
title('Testing error');
rmse=rms(error_test)
VAF=(1-var(yp-yphat)/var(yp))*100
%%
figure;
plot(yp)
hold on
plot(yphat,':','LineWidth',1.5)
xlim([1 1000])
title('For given test reference input')
legend('actual plant output','identification model output');