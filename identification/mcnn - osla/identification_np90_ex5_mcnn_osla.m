%%%% IDENTIFICATION EXAMPLE 5 OSLA

%%%%%%%%%%%%% IDENTIFICATION PROCESS
clc;clear;
k=1:50000;
f1=@(x,y)(x/(1+y^2));
f2=@(x,y)(x*y/(1+y^2));

yp=[zeros(2,1) zeros(2,length(k))];
yphat=[zeros(2,1) zeros(2,length(k))];
u=-1+2*randn(2,length(k)+1);

NN_1class=[2 200 1];
NN_2class=[2 200 1];
in_1=NN_1class(1);n1_1=NN_1class(2);out_1=NN_1class(3);
in_2=NN_2class(1);n1_2=NN_2class(2);out_2=NN_2class(3);

W1_1=randn(in_1+1,n1_1);
W2_1=zeros(n1_1+1,out_1);

W1_2=randn(in_2+1,n1_2);
W2_2=zeros(n1_2+1,out_2);

error_1_train=zeros(1,length(k));
error_2_train=zeros(1,length(k));

e1_1=0.001;
e2_1=0.5;
e1_2=0.01;
e2_2=1.0;

lambda=0.5;
P_1=(1/lambda)*eye(n1_1+1);
P_2=(1/lambda)*eye(n1_2+1);

for i=2:length(k)+1
    i
    f1x=f1(yp(1,i-1),yp(2,i-1));
    f2x=f2(yp(1,i-1),yp(2,i-1));
    yp(:,i)=[ f1x ; f2x]+u(:,i-1);
    
    A1_1=[1 yp(1,i-1) yp(2,i-1)]*W1_1;
    y1_1=tanh(A1_1);
    v1_1=[1 y1_1];
    A2_1=v1_1*W2_1;
    N_1=A2_1;
    
    A1_2=[1 yp(1,i-1) yp(2,i-1)]*W1_2;
    y1_2=tanh(A1_2);
    v1_2=[1 y1_2];
    A2_2=v1_2*W2_2;
    N_2=A2_2;
    
    yphat(:,i)=[N_1; N_2]+u(:,i-1);
    
    e=-(yphat(:,i)-yp(:,i));
    
    if(abs(e(1))>e1_1 && abs(e(1))<e2_1)
        P_1=P_1 - ((P_1*(v1_1')*v1_1*P_1)./(1+v1_1*P_1*v1_1'));
        W2_1=W2_1+e(1)*P_1*v1_1';
    elseif(abs(e(1))>e2_1)
        W1_1=[W1_1 randn(in_1+1,1)];
        W2_1=[W2_1;0];
        
        A1_1=[1 yp(1,i-1) yp(2,i-1)]*W1_1;
        y1_1=tanh(A1_1);
        v1_1=[1 y1_1];
        A2_1=v1_1*W2_1;
        N_1=A2_1;
        
        P_1=[P_1 zeros(size(P_1,1),1)];
        P_1=[P_1;zeros(1,size(P_1,2))];
        P_1(end,end)=1/lambda;
        P_1=P_1 - ((P_1*(v1_1')*v1_1*P_1)./(1+v1_1*P_1*v1_1'));
        W2_1=W2_1+e(1)*P_1*v1_1';
    end
    
    if(abs(e(2))>e1_2 && abs(e(2))<e2_2)
        P_2=P_2 - ((P_2*(v1_2')*v1_2*P_2)./(1+v1_2*P_2*v1_2'));
        W2_2=W2_2+e(2)*P_2*v1_2';
    elseif(abs(e(1))>e2_2)
        W1_2=[W1_2 randn(in_2+1,1)];
        W2_2=[W2_2;0];
        
        A1_2=[1 yp(1,i-1) yp(2,i-1)]*W1_2;
        y1_2=tanh(A1_2);
        v1_2=[1 y1_2];
        A2_2=v1_2*W2_2;
        N_2=A2_2;
        
        P_2=[P_2 zeros(size(P_2,1),1)];
        P_2=[P_2;zeros(1,size(P_2,2))];
        P_2(end,end)=1/lambda;
        P_2=P_2 - ((P_2*(v1_2')*v1_2*P_2)./(1+v1_2*P_2*v1_2'));
        W2_2=W2_2+e(1)*P_2*v1_2';
    end
end

k=1:10000;
yp=[zeros(2,1) zeros(2,length(k))];
yphat=[zeros(2,1) zeros(2,length(k))];
u=[sin((2*pi).*k./25);cos((2*pi).*k./25)];
error_1_test=zeros(1,length(k));
error_2_test=zeros(1,length(k));
for i=2:length(k)+1
    f1x=f1(yp(1,i-1),yp(2,i-1));
    f2x=f2(yp(1,i-1),yp(2,i-1));
    yp(:,i)=[ f1x ; f2x]+u(:,i-1);
    
    A1_1=[1 yphat(1,i-1) yphat(2,i-1)]*W1_1;
    y1_1=tanh(A1_1);
    A2_1=[1 y1_1]*W2_1;
    N_1=A2_1;
    
    A1_2=[1 yphat(1,i-1) yphat(2,i-1)]*W1_2;
    y1_2=tanh(A1_2);
    A2_2=[1 y1_2]*W2_2;
    N_2=A2_2;
    
    e_1= -(N_1 - f1x);
    e_2= -(N_2 - f2x);
    
    yphat(:,i)=[N_1; N_2]+u(:,i-1);
    
    error_1_test(i-1)=e_1;
    error_2_test(i-1)=e_2;
end
%%
figure;
subplot(121);
plot(error_1_test)
title('Testing error of NN1')
rmse_1=rms(error_1_test)
subplot(122);
plot(error_2_test)
title('Testing error of NN2')
rmse_2=rms(error_2_test)
VAF1=(1-var(yp(1,:)-yphat(1,:))/var(yp(1,:)))*100
VAF2=(1-var(yp(2,:)-yphat(2,:))/var(yp(2,:)))*100

%%
figure;
subplot(121);
plot(yp(1,:),'Linewidth',1.5);
hold on
plot(yphat(1,:),'Linewidth',1.5)
xlim([100 200])
title('response of actual plant and identification model(testing) for output 1');
xlabel('yp1 and yphat1')
legend('actual plant output','identification model output');
subplot(122);
plot(yp(2,:),'Linewidth',1.5);
hold on
plot(yphat(2,:),'Linewidth',1.5);
xlim([100 200])
title('response of actual plant and identification model(testing) for output 2');
xlabel('yp2 and yphat2')
legend('actual plant output','identification model output');