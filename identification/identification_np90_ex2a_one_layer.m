%%%% IDENTIFICATION EXAMPLE 2 PART A ONE LAYER

%%%% EQUILIBRIUM STATES OF UNFORCED SYSTEM
clc;clear;
k=[1:200];
f=@(x,y)(x*y*(x+2.5)/(1+x^2+y^2));
difftanh=@(x)(sech(x));

NNclass=[2 20 1];
in=NNclass(1);n1=NNclass(2);out=NNclass(3);

W1=normrnd(0,0.3,in+1,n1);                              %%%% Weight Initialization
W2=zeros(n1+1,out);

eta=0.1;                                                %%%% Learning Rate

points=1000
b=normrnd(0,4,1,points);                                   %%%% Initial conditions seed
a=normrnd(0,4,1,points);
a_zero=[];b_zero=[];
a_two=[];b_two=[];
a_nah=[];b_nah=[];
subplot(121);
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
        N=A2;
        % Identification Model Ouput
        yphat(i)=N+u;
        % Backward Pass
        e=-(yphat(i)-yp(i));
        del2=e;
        del1=difftanh(A1).*(del2*W2(2:end,:)');
        Jw2=[1 y1]'*del2;                               %%%% CRAY-DIENTS
        Jw1=[1 yp(i-1) yp(i-2)]'*del1;
        %Weight Updation
        W1=W1+eta*Jw1;
        W2=W2+eta*Jw2;
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
    plot(yp(1:60))
    hold on
end
title('Plant output for different initial conditions');
subplot(122);
scatter(a_zero,b_zero,'b','o');
hold on;
scatter(a_two,b_two,'r','x');
hold on;
scatter(a_nah,b_nah,'g','d');
title('Initial conditions leading to the two equilibrium states(plant)');
legend('eq state=0','eq state=2','not eq state');
%%
%Performance Measure
%%
%Saving Weights
save("WeightsH1_NP90_ex2a_one_layer",'W1','W2')
disp('saved')
%%
%Loading Weights
clear;
load("WeightsH1_NP90_ex2a_one_layer")
disp('loaded')
%%
figure;
k=[1:200];
f=@(x,y)(x*y*(x+2.5)/(1+x^2+y^2));
%b=1;a=2;
points=1000;
b=normrnd(0,4,1,points);
a=normrnd(0,4,1,points);
a_zero=[];b_zero=[];
a_two=[];b_two=[];
a_nah=[];b_nah=[];
subplot(121);
for j=1:length(b)
    yp=[b(j) a(j) zeros(1,length(k))];
    yphat=[b(j) a(j) zeros(1,length(k))];
    for i=3:length(k)+2
        u=0;
        yp(i)=f(yp(i-1),yp(i-2))+u;
        A1=[1 yphat(i-1) yphat(i-2)]*W1;
        y1=tanh(A1);
        A2=[1 y1]*W2;
        N=A2;
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
    plot(yphat(1:60))
    hold on
end
title('Identification Model output for different initial conditions');
subplot(122);
scatter(a_zero,b_zero,'b','o');
hold on
scatter(a_two,b_two,'r','x');hold on;
scatter(a_nah,b_nah,'g','d');
title('Initial conditions leading to the two equilibrium states(identification model)');
legend('eq state=0','eq state=2','not eq state');

