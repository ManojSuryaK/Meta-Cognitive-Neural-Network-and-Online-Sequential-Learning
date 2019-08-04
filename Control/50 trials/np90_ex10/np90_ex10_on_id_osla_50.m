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
for q=1:50
    q
    
k=1:50000;

ec=zeros(2,length(k)+2);

yp=[zeros(2,1) zeros(2,length(k))];
ym=[zeros(2,1) zeros(2,length(k))];
yphat=[zeros(2,1) zeros(2,length(k))];

NN_1class=[2 50 1];
NN_2class=[2 50 1];
in_1=NN_1class(1);n1_1=NN_1class(2);out_1=NN_1class(3);
in_2=NN_2class(1);n1_2=NN_2class(2);out_2=NN_2class(3);

W1_1=rand(in_1+1,n1_1);
W2_1=zeros(n1_1+1,out_1);

W1_2=rand(in_2+1,n1_2);
W2_2=zeros(n1_2+1,out_2);

lambda=1e-02;
P_1=(1/lambda)*eye(n1_1+1);
P_2=(1/lambda)*eye(n1_2+1);

uc=0;
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
    
    P_1=P_1 - ((P_1*(v1_1')*v1_1*P_1)./(1+v1_1*P_1*v1_1'));
    W2_1=W2_1+e(1)*P_1*v1_1';
    
    P_2=P_2 - ((P_2*(v1_2')*v1_2*P_2)./(1+v1_2*P_2*v1_2'));
    W2_2=W2_2+e(2)*P_2*v1_2';
    
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
absol_ec1=abs(ec(1,:));
error_thresh1=1.0;
ind1=find(absol_ec1<error_thresh1);
conv_time1=0;
for i=1:length(ind1)
    z=diff(ind1(i:i+100))==1;
    if sum(z)==100
        conv_time1=ind1(i);
        break;
    else
        x=find(z==0);
        i=x(end);
    end
end
%%
absol_ec2=abs(ec(2,:));
error_thresh2=1.0;
ind2=find(absol_ec2<error_thresh2);
conv_time2=0;
for i=1:length(ind2)
    z=diff(ind2(i:i+100))==1;
    if sum(z)==100
        conv_time2=ind2(i);
        break;
    else
        x=find(z==0);
        i=x(end);
    end
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
save('ex10_50_osla','allrmse1','allVAF1','allconvtime1','allneurons1','allrmse2','allVAF2','allconvtime2','allneurons2');
end
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