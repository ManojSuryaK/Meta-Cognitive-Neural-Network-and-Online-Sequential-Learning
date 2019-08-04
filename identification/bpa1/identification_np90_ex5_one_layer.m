%%%% IDENTIFICATION EXAMPLE 5 ONE LAYER

%%%%%%%%%%%%% IDENTIFICATION PROCESS
clc;clear;
k=1:50000;
f1=@(x,y)(x/(1+y^2));
f2=@(x,y)(x*y/(1+y^2));
difftanh=@(x)(sech(x).^2);

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

eta=0.001;

for i=2:length(k)+1
    f1x=f1(yp(1,i-1),yp(2,i-1));
    f2x=f2(yp(1,i-1),yp(2,i-1));
    yp(:,i)=[ f1x ; f2x]+u(:,i-1);
    
    A1_1=[1 yp(1,i-1) yp(2,i-1)]*W1_1;
    y1_1=tanh(A1_1);
    A2_1=[1 y1_1]*W2_1;
    N_1=A2_1;
    
    A1_2=[1 yp(1,i-1) yp(2,i-1)]*W1_2;
    y1_2=tanh(A1_2);
    A2_2=[1 y1_2]*W2_2;
    N_2=A2_2;
    
    e_1= -(N_1 - f1x);
    e_2= -(N_2 - f2x);
    
    del2_1=e_1;
    del1_1=difftanh(A1_1).*(del2_1*W2_1(2:end,:)');
    Jw2_1=[1 y1_1]'*del2_1;
    Jw1_1=[1 yp(1,i-1) yp(2,i-1)]'*del1_1;
    W2_1=W2_1+eta*Jw2_1;
    W1_1=W1_1+eta*Jw1_1;
    
    del2_2=e_2;
    del1_2=difftanh(A1_2).*(del2_2*W2_2(2:end,:)');
    Jw2_2=[1 y1_2]'*del2_2;
    Jw1_2=[1 yp(1,i-1) yp(2,i-1)]'*del1_2;
    W2_2=W2_2+eta*Jw2_2;
    W1_2=W1_2+eta*Jw1_2;
    
    yphat(:,i)=[N_1; N_2]+u(:,i-1);
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

subplot(121);
plot(yp(1,:));
hold on
plot(yphat(1,:))
xlim([100 200])
title('response of actual plant and identification model(testing) for output 1');
xlabel('yp1 and yphat1')
legend('actual plant output','identification model output');
subplot(122);
plot(yp(2,:));
hold on
plot(yphat(2,:));
xlim([100 200])
title('response of actual plant and identification model(testing) for output 2');
xlabel('yp2 and yphat2')
legend('actual plant output','identification model output');