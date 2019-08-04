clc;clear;

allrmse=[];
allVAF=[];
allconvtime=[];
allneurons=[];
for q=1:50
    q
    
k=1:50000;

ec=zeros(1,length(k)+2);

yp=[0 0 0 zeros(1,length(k))];
ym=[0 0 0 zeros(1,length(k))];
yphat=[0 0 0 zeros(1,length(k))];

NNclass=[3 70 1];
in=NNclass(1);n1=NNclass(2);out=NNclass(3);

W1=randn(in+1,n1);
W2=zeros(n1+1,out);

eta=0.005;

e1=0.001;
e2=2.5;

uc=[0 zeros(1,length(k))];
A1=[1 yp(3) yp(2) yp(1)]*W1;
y1=tanh(A1);
A2=[1 y1]*W2;
N=A2;
for i=4:length(k)+3
    r=sin(2*pi*(i-4)/25);
    ym(i)=0.32*ym(i-1)+0.64*ym(i-2)-0.5*ym(i-3)+r;

    f=5*yp(i-1)*yp(i-2)/(1+yp(i-1)^2+yp(i-2)^2+yp(i-3)^2);
    yp(i)=f+uc(i-2)+0.8*uc(i-3);
    yphat(i)=N+uc(i-2)+0.8*uc(i-3);
    
    ec(i)=-(yp(i)-ym(i));
    e=-(yphat(i)-yp(i));

    if(e1<abs(e) && abs(e)<e2)
        del2=e;
        del1=(1-y1.^2).*(del2*W2(2:end,:)');
        Jw2=[1 y1]'*del2;
        Jw1=[1 yp(i-1) yp(i-2) yp(i-3)]'*del1;
        W2=W2+eta*Jw2;
        W1=W1+eta*Jw1;
    elseif(abs(e)>e2)
        W1=[W1 randn(in+1,1)];
        W2=[W2;0];
        
        A1=[1 yp(i-1) yp(i-2) yp(i-3)]*W1;
        y1=tanh(A1);
        A2=[1 y1]*W2;
        N=A2;
        del2=e;
        del1=(1-y1.^2).*(del2*W2(2:end,:)');
        Jw2=[1 y1]'*del2;         
        Jw1=[1 yp(i-1) yp(i-2) yp(i-3)]'*del1;
        W2=W2+eta*Jw2;
        W1=W1+eta*Jw1;
        n1=n1+1;
    end
    A1=[1 yp(i) yp(i-1) yp(i-2)]*W1;
    y1=tanh(A1);
    A2=[1 y1]*W2;
    N=A2;
    
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
    disp('trial deleted');
    disp(q)
    continue;
end
%%
allrmse(q)=rmse;
allVAF(q)=VAF;
allconvtime(q)=conv_time;
allneurons(q)=n1;
end
save('ex8_50_mcnn_bpa','allrmse','allVAF','allconvtime','allneurons');
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