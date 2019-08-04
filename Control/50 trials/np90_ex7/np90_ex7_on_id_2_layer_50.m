%%%% CONTROL EXAMPLE 7 SIMULTANEOUS
clc;clear;

allrmse=[];
allVAF=[];
allconvtime=[];
allneurons=[];
for q=1:50
    q
    
k=1:50000;

ec=zeros(1,length(k)+2);

yp=[0 0 zeros(1,length(k))];
yphat=[0 0 zeros(1,length(k))];
ym=[0 0 zeros(1,length(k))];

NNclass=[2 20 10 1];
in=NNclass(1);n1=NNclass(2);n2=NNclass(3);out=NNclass(4);

W1=randn(in+1,n1);
W2=randn(n1+1,n2);
W3=zeros(n2+1,out);

eta=0.1;

u1=0;
A1=[1 yp(2) yp(1)]*W1;
y1=tanh(A1);
A2=[1 y1]*W2;
y2=tanh(A2);
A3=[1 y2]*W3;
N=A3;

for i=3:length(k)+2
    r=sin(2*pi*(i-3)/25);
    ym(i)=0.6*ym(i-1)+0.2*ym(i-2)+r;
    
    f=yp(i-1)*yp(i-2)*(yp(i-1)+2.5)/(1+yp(i-1)^2+yp(i-2)^2);
    yp(i)=f+u1;
    yphat(i)=N+u1;
    
    ec(i)=-(yp(i)-ym(i));
    e=-(yphat(i)-yp(i));
    del3=e;
    del2=(1-y2.^2).*(del3*W3(2:end,:)');
    del1=(1-y1.^2).*(del2*W2(2:end,:)');
    Jw3=[1 y2]'*del3;
    Jw2=[1 y1]'*del2;
    Jw1=[1 yp(i-1) yp(i-2)]'*del1;
    W3=W3+eta*Jw3;
    W2=W2+eta*Jw2;
    W1=W1+eta*Jw1;
    
    A1=[1 yp(i) yp(i-1)]*W1;
    y1=tanh(A1);
    A2=[1 y1]*W2;
    y2=tanh(A2);
    A3=[1 y2]*W3;
    N=A3;

    r=sin(2*pi*(i-2)/25);
    u1=-N+0.6*yp(i)+0.2*yp(i-1)+r;    
end
%%
rmse=rms(ec);
VAF=(1-var(ym-yp)/var(ym))*100;
%%
%{
plot(ym)
hold on
plot(yp)
xlim([0 200])
figure;
plot(ec)
xlim([0 200])
%%
save('C:\Users\tarun\Documents\MATLAB\NP90\control\plots_for_paper\graph_ex1_bpa2','ym','yp','ec')
disp('saved')
%}
%%
absol_ec=abs(ec);
error_thresh=1.0;
ind=find(absol_ec<error_thresh);
conv_time=0;
flag=0;
for i=1:length(ind)
    if ((i+100)>length(ind))
        flag=1;
        break;
    end
    z=diff(ind(i:i+100))==1;
    if sum(z)==100
        conv_time=ind(i);
        break;
    else
        x=find(z==0);
        i=x(end);
    end
end
if(flag==1)
    q=q-1;
    disp('trial deleted');
    continue;
end
%%
allrmse(q)=rmse;
allVAF(q)=VAF;
allconvtime(q)=conv_time;
allneurons(q)=n1;
end
save('ex7_50_bpa2','allrmse','allVAF','allconvtime');
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