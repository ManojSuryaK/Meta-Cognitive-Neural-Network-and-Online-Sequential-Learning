%%%% IDENTIFICATION EXAMPLE 2 PART A

%%%% EQUILIBRIUM STATES OF UNFORCED SYSTEM
clc;clear;
k=[1:100000];
f=@(x,y)(x*y*(x+2.5)/(1+x^2+y^2));
difftanh=@(x)(sech(x));

NNclass=[2 20 10 1];
in=NNclass(1);n1=NNclass(2);n2=NNclass(3);out=NNclass(4);

W1=randn(in+1,n1);                              %%%% Weight Initialization
W2=randn(n1+1,n2);
W3=zeros(n2+1,out);

eta=0.05;                                                %%%% Learning Rate
points=1000
b=normrnd(0,4,1,points);                                   %%%% Initial conditions seed
a=normrnd(0,4,1,points);
a_zero=[];b_zero=[];
a_two=[];b_two=[];
a_nah=[];b_nah=[];
error=zeros(1,length(k)+2);
for j=1:length(b)                                       %%%% Training the network to different initial conditions
    yp=[b(j) a(j) zeros(1,length(k))];                  
    yphat=[b(j) a(j) zeros(1,length(k))];

    for i=3:length(k)+2
        u=0;
        yp(i)=f(yp(i-1),yp(i-2))+u;
        %%% NEURAL NETWORK
        % Forward Pass
        A1=[1 yp(i-1) yp(i-2)]*W1;
        y1=tanh(A1);
        A2=[1 y1]*W2;
        y2=tanh(A2);
        A3=[1 y2]*W3;
        N=A3;
        % Identification Model Ouput
        yphat(i)=N+u;
        % Backward Pass
        e=-(yphat(i)-yp(i));
        del3=e;
        del2=difftanh(A2).*(del3*W3(2:end,:)');
        del1=difftanh(A1).*(del2*W2(2:end,:)');
        Jw3=[1 y2]'*del3;                               %%%% CRAY-DIENTS
        Jw2=[1 y1]'*del2;
        Jw1=[1 yp(i-1) yp(i-2)]'*del1;
        %Weight Updation
        W1=W1+eta*Jw1;
        W2=W2+eta*Jw2;
        W3=W3+eta*Jw3;
        error(i)=e;
    end
    if(yphat(end)<0.1 && yphat(end)>-0.1)               %%%% depicting initial conditions
        a_zero=[a_zero a(j)];                           %%%% which lead to the two equilibrium states
        b_zero=[b_zero b(j)];
    elseif(yp(end)<2.1 && yp(end)>1.8)
        a_two=[a_two a(j)];
        b_two=[b_two b(j)];
    else
        a_nah=[a_nah a(j)];
        b_nah=[b_nah b(j)];
    end
    %plot(yp(1:60))
    %hold on
end
%title('Plant output for different initial conditions');
%subplot(122);
scatter(a_zero,b_zero,'b','o');
hold on;
scatter(a_two,b_two,'r','x');
hold on;
scatter(a_nah,b_nah,'g','d');
title('Initial conditions leading to the two equilibrium states(plant)');
legend('eq state=0','eq state=2','not eq state');
%%
%Performance Measure
%{
figure;
plot(error)
title('error')
rmse1=rms(error(1:10000));
rmse2=rms(error(10001:end));
fprintf('RMS error 1 = %f\n',rmse1);
fprintf('RMS error 2 = %f\n\n',rmse2);
%}
%%
%Saving Weights
save("WeightsH2_NP90_ex2a",'W1','W2','W3')
disp('saved')
%%
%Loading Weights
clear;
load("WeightsH2_NP90_ex2a")
disp('loaded')
%%
figure;
k=[1:100];
f=@(x,y)(x*y*(x+2.5)/(1+x^2+y^2));
%b=1;a=2;
points=1000;
b=normrnd(0,4,1,points);
a=normrnd(0,4,1,points);
a_zero=[];b_zero=[];
a_two=[];b_two=[];
a_nah=[];b_nah=[];
for j=1:length(b)
    yp=[b(j) a(j) zeros(1,length(k))];
    yphat=[b(j) a(j) zeros(1,length(k))];
    for i=3:length(k)+2
        u=0;
        yp(i)=f(yp(i-1),yp(i-2))+u;
        A1=[1 yphat(i-1) yphat(i-2)]*W1;
        y1=tanh(A1);
        A2=[1 y1]*W2;
        y2=tanh(A2);
        A3=[1 y2]*W3;
        N=A3;
        yphat(i)=N+u;
    end
    if(yphat(end)<=0.1 && yphat(end)>=-0.1)
        a_zero=[a_zero a(j)];
        b_zero=[b_zero b(j)];
    elseif(yphat(end)<=2.2 && yphat(end)>=1.8)
        a_two=[a_two a(j)];
        b_two=[b_two b(j)];
    else
        a_nah=[a_nah a(j)];
        b_nah=[b_nah b(j)];
    end
    %plot(yphat(1:60))
    %hold on
end
%title('Identification Model output for different initial conditions');
%subplot(122);
scatter(a_zero,b_zero,'b','o');
hold on
scatter(a_two,b_two,'r','x');hold on;
scatter(a_nah,b_nah,'g','d');
title('Initial conditions leading to the two equilibrium states(identification model)');
legend('eq state=0','eq state=2','not eq state');

