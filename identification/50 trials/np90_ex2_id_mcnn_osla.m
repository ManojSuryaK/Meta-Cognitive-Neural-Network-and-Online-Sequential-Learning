clc;clear;

allrmse=[];
allVAF=[];
allconvtime=[];
allneurons=[];
for q=1:50
    q
    
k=[1:50000];
f=@(x,y)(x*y*(x+2.5)/(1+x^2+y^2));

NNclass=[2 50 1];
in=NNclass(1);n1=NNclass(2);out=NNclass(3);

W1=randn(in+1,n1);
W2=zeros(n1+1,out);

e1=0.001;
e2=0.50;

lambda=1e-04;
P=(1/lambda)*eye(n1+1);        

yp=[0 0 zeros(1,length(k))];                  
yphat=[0 0 zeros(1,length(k))];

u=(-2+4*rand(1,length(k)+2));
for i=3:length(k)+2
    yp(i)=f(yp(i-1),yp(i-2))+u(i-1);
   
    A1=[1 yp(i-1) yp(i-2)]*W1;
    y1=tanh(A1);
    v1=[1 y1];
    A2=v1*W2;
    N=A2;
    
    yphat(i)=N+u(i-1);
    e=-(yphat(i)-yp(i));
    
    if(abs(e)<e1)
        continue;
    elseif(abs(e)>e1 && abs(e)<e2)    
        P=P - ((P*v1'*v1*P)./(1+v1*P*v1'));
        W2=W2+e*P*v1';
    else
        W1=[W1 randn(in+1,1)];
        W2=[W2;0];
        
        A1=[1 yp(i-1) yp(i-2)]*W1;
        y1=tanh(A1);
        v1=[1 y1];
        A2=[1 y1]*W2;
        N=A2;
        P=[P zeros(size(P,1),1)];
        P=[P;zeros(1,size(P,2))];
        P(end,end)=1/lambda;
        P=P - ((P*v1'*v1*P)./(1+v1*P*v1'));
        W2=W2+e*P*v1';
        n1=n1+1;
    end
end

yp=[0 0 zeros(1,length(k))];                  
yphat=[0 0 zeros(1,length(k))];
ei=zeros(1,length(k));
for i=3:length(k)+2
    u=sin(2*pi*(i-3)/25);
    yp(i)=f(yp(i-1),yp(i-2))+u;
    
    A1=[1 yphat(i-1) yphat(i-2)]*W1;
    y1=tanh(A1);
    A2=[1 y1]*W2;
    N=A2;
    
    yphat(i)=N+u;
    e=-(yphat(i)-yp(i));
    ei(i-2)=e;
end
%%
rmse=rms(ei);
VAF=(1-var(yp-yphat)/var(yp))*100;
%%
absol_ei=abs(ei);
error_thresh=0.25;
ind=find(absol_ei<error_thresh);
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
i=find(~allVAF)
allrmse(i)=[];
allVAF(i)=[];
allconvtime(i)=[];
allneurons(i)=[];
save('mcnn_osla_ex2_50','allrmse','allVAF','allconvtime','allneurons');
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