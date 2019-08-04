clc;clear;

allrmse=[];
allVAF=[];
allconvtime=[];
allneurons=[];
for q=1:20
    q
    
k=1:50000;

ec=zeros(1,length(k)+2);

yp=[0 0 0 zeros(1,length(k))];
ym=[0 0 0 zeros(1,length(k))];
yphat=[0 0 0 zeros(1,length(k))];

NNclass=[3 20 10 1];
in=NNclass(1);n1=NNclass(2);n2=NNclass(3);out=NNclass(4);

W1=randn(in+1,n1);
W2=randn(n1+1,n2);
W3=zeros(n2+1,out);

eta=0.05;

uc=[0 0 0 zeros(1,length(k))];
A1=[1 yp(3) yp(2) yp(1)]*W1;
y1=tanh(A1);
A2=[1 y1]*W2;
y2=tanh(A2);
A3=[1 y2]*W3;
N=A3;
for i=4:length(k)+3
    r=sin(2*pi*(i-4)/25);
    ym(i)=0.32*ym(i-1)+0.64*ym(i-2)-0.5*ym(i-3)+r;

    f=5*yp(i-1)*yp(i-2)/(1+yp(i-1)^2+yp(i-2)^2+yp(i-3)^2);
    yp(i)=f+uc(i-2)+0.8*uc(i-3);
    yphat(i)=N+uc(i-2)+0.8*uc(i-3);
    
    ec(i)=-(yp(i)-ym(i));
    e=-(yphat(i)-yp(i));
    del3=e;
    del2=(1-y2.^2).*(del3*W3(2:end,:)');
    del1=(1-y1.^2).*(del2*W2(2:end,:)');
    Jw3=[1 y2]'*del3;
    Jw2=[1 y1]'*del2;
    Jw1=[1 yp(i-1) yp(i-2) yp(i-3)]'*del1;
    W3=W3+eta*Jw3;
    W2=W2+eta*Jw2;
    W1=W1+eta*Jw1;
    
    A1=[1 yp(i) yp(i-1) yp(i-2)]*W1;
    y1=tanh(A1);
    A2=[1 y1]*W2;
    y2=tanh(A2);
    A3=[1 y2]*W3;
    N=A3;
    
    r=sin(2*pi*(i-3)/25);
    uc(i-1)=-N-0.8*uc(i-2)+0.32*yp(i)+0.64*yp(i-1)-0.5*yp(i-2)+r;
end
%%
rmse=rms(ec);
VAF=(1-var(ym-yp)/var(ym))*100;
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
save('ex8_50_bpa2','allrmse','allVAF','allconvtime','allneurons');
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