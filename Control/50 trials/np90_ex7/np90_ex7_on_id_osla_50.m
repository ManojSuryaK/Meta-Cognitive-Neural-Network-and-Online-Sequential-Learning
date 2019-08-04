%%%% CONTROL EXAMPLE 7 SIMULTANEOUS OSLA
clc;clear;

allrmse=[];
allVAF=[];
allconvtime=[];
allneurons=[];
for q=1:1
    q
    
k=1:50000;

ec=zeros(1,length(k));

yp=[0 0 zeros(1,length(k))];
yphat=[0 0 zeros(1,length(k))];
ym=[0 0 zeros(1,length(k))];

NNclass=[2 50 1];
in=NNclass(1);n1=NNclass(2);out=NNclass(3);

W1=randn(in+1,n1);
W2=zeros(n1+1,out);

lambda=1e-03;
P=(1/lambda)*eye(n1+1);

u1=0;
A1=[1 yp(2) yp(1)]*W1;
y1=tanh(A1);
v1=[1 y1];
A2=v1*W2;
N=A2;
for i=3:length(k)+2
    r=sin(2*pi*(i-3)/25);
    ym(i)=0.6*ym(i-1)+0.2*ym(i-2)+r;
    
    f=yp(i-1)*yp(i-2)*(yp(i-1)+2.5)/(1+yp(i-1)^2+yp(i-2)^2);
    yp(i)=f+u1;
    yphat(i)=N+u1;
    
    ec(i-2)=-(yp(i)-ym(i));
    e=-(yphat(i)-yp(i));
    
    P=P - ((P*v1'*v1*P)./(1+v1*P*v1'));
    W2=W2+e*P*v1';
    
    A1=[1 yp(i) yp(i-1)]*W1;
    y1=tanh(A1);
    v1=[1 y1];
    A2=v1*W2;
    N=A2;
    
    u1=-N+0.6*yp(i)+0.2*yp(i-1)+r;
end
%%
if(isnan(A2))
    q=q-1;
    disp('trial deleted');
    continue;
end
%%
rmse=rms(ec);
VAF=(1-var(ym-yp)/var(ym))*100;
%%
plot(ym)
hold on
plot(yp)
xlim([0 200])
figure;
plot(ec)
xlim([0 200])
%%
save('C:\Users\tarun\Documents\MATLAB\NP90\control\plots_for_paper\graph_ex1_osla','ym','yp','ec')
disp('saved')
%%
absol_ec=abs(ec);
error_thresh=1.0;
ind=find(absol_ec<error_thresh);
conv_time=0;
for i=1:length(ind)
    z=diff(ind(i:i+100))==1;
    if sum(z)==100
        conv_time=ind(i);
        break;
    else
        x=find(z==0);
        i=x(end);
    end
end
%%
allrmse(q)=rmse;
allVAF(q)=VAF;
allconvtime(q)=conv_time;
allneurons(q)=n1;
end
save('ex7_50_osla','allrmse','allVAF','allconvtime');
%%
avgrmse=mean(allrmse)
stdrmse=std(allrmse)
avgVAF=mean(allVAF)
stdVAF=std(allVAF)

avgconvtime=mean(allconvtime)
minconvtime=min(allconvtime)
maxconvtime=max(allconvtime)

avgneurons=mean(allneurons)
minneurons=min(allneurons)
maxneurons=max(allneurons)