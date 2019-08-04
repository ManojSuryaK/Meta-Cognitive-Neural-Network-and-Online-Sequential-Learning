%%%%%%% CONTROL PROBLEM 10 
clc;clear;

allrmse1=[];
allVAF1=[];
allconvtime1=[];
allneurons1=[];
allrmse2=[];
allVAF2=[];
allconvtime2=[];
allneurons2=[];
qdel=[];
for q=1:1
    q
    
k=1:50000;

ec=zeros(2,length(k)+2);

yp=[zeros(2,1) zeros(2,length(k))];
ym=[zeros(2,1) zeros(2,length(k))];
yphat=[zeros(2,1) zeros(2,length(k))];

NN_1class=[2 40 1];
NN_2class=[2 40 1];
in_1=NN_1class(1);n1_1=NN_1class(2);out_1=NN_1class(3);
in_2=NN_2class(1);n1_2=NN_2class(2);out_2=NN_2class(3);

W1_1=randn(in_1+1,n1_1);
W2_1=zeros(n1_1+1,out_1);

W1_2=randn(in_2+1,n1_2);
W2_2=zeros(n1_2+1,out_2);

e1_1=0.001;
e2_1=0.5;
e1_2=0.001;
e2_2=0.5;

lambda=1e-02;
P_1=(1/lambda)*eye(n1_1+1);
P_2=(1/lambda)*eye(n1_2+1);

uc=[0;0];
A1_1=[1 yp(1,1) yp(2,1)]*W1_1;
y1_1=tanh(A1_1);
v1_1=[1 y1_1];
A2_1=v1_1*W2_1;
N_1=A2_1;

A1_2=[1 yp(1,1) yp(2,1)]*W1_2;
y1_2=tanh(A1_2);
v1_2=[1 y1_2];
A2_2=v1_2*W2_2;
N_2=A2_2;
for i=2:length(k)+1
    r=[sin(2*pi*(i-2)/25);cos(2*pi*(i-2)/25)];
    ym(:,i)=[0.6 0.2; 0.1 -0.8]*ym(:,i-1)+r;

    f=[yp(1,i-1)/(1+yp(2,i-1)^2) ; yp(1,i-1)*yp(2,i-1)/(1+yp(2,i-1)^2)];
    yp(:,i)=f+uc;
    yphat(:,i)=[N_1;N_2]+uc;
    
    ec(:,i)=-(yp(:,i)-ym(:,i));
    e=-(yphat(:,i)-yp(:,i));
    
    if(e1_1<abs(e(1)) && abs(e(1))<e2_1)
        P_1=P_1 - ((P_1*(v1_1')*v1_1*P_1)./(1+v1_1*P_1*v1_1'));
        W2_1=W2_1+e(1)*P_1*v1_1';
    elseif(abs(e(1))>e2_1)
        W1_1=[W1_1 randn(in_1+1,1)];
        W2_1=[W2_1;0];
        P_1=[P_1 zeros(size(P_1,1),1)];
        P_1=[P_1;zeros(1,size(P_1,2))];
        P_1(end,end)=1/lambda;
        
        A1_1=[1 yp(1,i-1) yp(2,i-1)]*W1_1;
        y1_1=tanh(A1_1);
        v1_1=[1 y1_1];
        A2_1=v1_1*W2_1;
        N_1=A2_1;

        P_1=P_1 - ((P_1*(v1_1')*v1_1*P_1)./(1+v1_1*P_1*v1_1'));
        W2_1=W2_1+e(1)*P_1*v1_1';
        n1_1=n1_1+1;
    end
    
    if(e1_2<abs(e(2)) && abs(e(2))<e2_2)
        P_2=P_2 - ((P_2*(v1_2')*v1_2*P_2)./(1+v1_2*P_2*v1_2'));
        W2_2=W2_2+e(2)*P_2*v1_2';
    elseif(abs(e(2))>e2_2)
        W1_2=[W1_2 randn(in_2+1,1)];
        W2_2=[W2_2;0];
        P_2=[P_2 zeros(size(P_2,1),1)];
        P_2=[P_2;zeros(1,size(P_2,2))];
        P_2(end,end)=1/lambda;
        
        A1_2=[1 yp(1,i-1) yp(2,i-1)]*W1_2;
        y1_2=tanh(A1_2);
        v1_2=[1 y1_2];
        A2_2=v1_2*W2_2;
        N_2=A2_2;

        P_2=P_2 - ((P_2*(v1_2')*v1_2*P_2)./(1+v1_2*P_2*v1_2'));
        W2_2=W2_2+e(2)*P_2*v1_2';
        n1_2=n1_2+1;
    end    
    A1_1=[1 yp(1,i) yp(2,i)]*W1_1;
    y1_1=tanh(A1_1);
    v1_1=[1 y1_1];
    A2_1=v1_1*W2_1;
    N_1=A2_1;
    
    A1_2=[1 yp(1,i) yp(2,i)]*W1_2;
    y1_2=tanh(A1_2);
    v1_2=[1 y1_2];
    A2_2=v1_2*W2_2;
    N_2=A2_2;
    
    uc=-[N_1;N_2]+[0.6 0.2; 0.1 -0.8]*yp(:,i)+r;
end
%%
rmse1=rms(ec(1,:));
rmse2=rms(ec(2,:));
VAF1=(1-var(ym(1,:)-yp(1,:))/var(ym(1,:)))*100;
VAF2=(1-var(ym(2,:)-yp(2,:))/var(ym(2,:)))*100;
%%
%{
n=00;
subplot(221);
plot(ym(1,:))
hold on
plot(yp(1,:))
xlim([n  n+200])
legend('reference model output','controlled plant output');
title('Controlled plant response yp1')
ylabel('ym1 and yp1')
subplot(222);
plot(ym(2,:))
hold on
plot(yp(2,:))
xlim([n n+200])
title('Controlled plant response yp2')
ylabel('ym2 and yp2')
legend('reference model output','controlled plant output');
subplot(223);
plot(ec(1,:))
title('Control Error yp1')
xlabel('time steps')
ylabel('ec1')
xlim([n  n+200])
subplot(224);
plot(ec(2,:))
title('Control Error yp2')
xlabel('time steps')
ylabel('ec2')
xlim([n  n+200])
%}
%%
save('C:\Users\tarun\Documents\MATLAB\NP90\control\plots_for_paper\graph_ex3_mcnn_osla','ym','yp','ec')
disp('saved')
%%
absol_ec1=abs(ec(1,:));
error_thresh1=1.0;
ind1=find(absol_ec1<error_thresh1);
conv_time1=0;
flag=0;
for i=1:length(ind1)
    if ((i+100)>length(ind1))
        flag=1;
        break;
    end
    z=diff(ind1(i:i+100))==1;
    if sum(z)==100
        conv_time1=ind1(i);
        break;
    else
        x=find(z==0);
        i=x(end);
    end
end
if(flag==1)
    qdel=[qdel q];
    disp('trial deleted');
    continue;
end
%%
absol_ec2=abs(ec(2,:));
error_thresh2=1.0;
ind2=find(absol_ec2<error_thresh2);
conv_time2=0;
for i=1:length(ind2)
    if ((i+100)>length(ind2))
        flag=1;
        break;
    end
    z=diff(ind2(i:i+100))==1;
    if sum(z)==100
        conv_time2=ind2(i);
        break;
    else
        x=find(z==0);
        i=x(end);
    end
end
if(flag==1)
    qdel=[qdel q];
    disp('trial deleted');
    continue;
end
%%
allrmse1(q)=rmse1;
allVAF1(q)=VAF1;
allconvtime1(q)=conv_time1;
allneurons1(q)=n1_1;

allrmse2(q)=rmse2;
allVAF2(q)=VAF2;
allconvtime2(q)=conv_time2;
allneurons2(q)=n1_2;
save('ex10_50_mcnn_osla','allrmse1','allVAF1','allconvtime1','allneurons1','allrmse2','allVAF2','allconvtime2','allneurons2');
end
%%
id=find(~allVAF1)
allrmse1(id)=[];
allVAF1(id)=[];
allconvtime1(id)=[];
allneurons1(id)=[];

allrmse2(id)=[];
allVAF2(id)=[];
allconvtime2(id)=[];
allneurons2(id)=[];
%%
avgrmse1=mean(allrmse1)
stdrmse1=std(allrmse1)
avgVAF1=mean(allVAF1)
stdVAF1=std(allVAF1)

avgconvtime1=mean(allconvtime1)
minconvtime1=min(allconvtime1)
maxconvtime1=max(allconvtime1)

avgneurons1=mean(allneurons1)
minneurons1=min(allneurons1)
maxneurons1=max(allneurons1)

avgrmse2=mean(allrmse2)
stdrmse2=std(allrmse2)
avgVAF2=mean(allVAF2)
stdVAF2=std(allVAF2)

avgconvtime2=mean(allconvtime2)
minconvtime2=min(allconvtime2)
maxconvtime2=max(allconvtime2)

avgneurons2=mean(allneurons2)
minneurons2=min(allneurons2)
maxneurons2=max(allneurons2)