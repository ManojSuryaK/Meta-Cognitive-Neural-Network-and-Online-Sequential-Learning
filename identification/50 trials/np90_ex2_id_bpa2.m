clc;clear;

allrmse=[];
allVAF=[];
allconvtime=[];
allneurons=[];
for q=1:50
    q
    
k=[1:50000];
f=@(x,y)(x*y*(x+2.5)/(1+x^2+y^2));
difftanh=@(x)(sech(x).^2);

NNclass=[2 20 10 1];
in=NNclass(1);n1=NNclass(2);n2=NNclass(3);out=NNclass(4);

W1=randn(in+1,n1);
W2=randn(n1+1,n2);
W3=zeros(n2+1,out);

eta=0.01;          

yp=[0 0 zeros(1,length(k))];                  
yphat=[0 0 zeros(1,length(k))];

u=(-2+4*rand(1,length(k)+2));
for i=3:length(k)+2
    yp(i)=f(yp(i-1),yp(i-2))+u(i-1);
   
    A1=[1 yp(i-1) yp(i-2)]*W1;
    y1=tanh(A1);
    A2=[1 y1]*W2;
    y2=tanh(A2);
    A3=[1 y2]*W3;
    N=A3;
    
    yphat(i)=N+u(i-1);
    e=-(yphat(i)-yp(i));
    
    del3=e;
    del2=difftanh(A2).*(del3*W3(2:end,:)');
    del1=difftanh(A1).*(del2*W2(2:end,:)');
    Jw3=[1 y2]'*del3;
    Jw2=[1 y1]'*del2;
    Jw1=[1 yp(i-1) yp(i-2)]'*del1;
    
    W3=W3+eta*Jw3;
    W2=W2+eta*Jw2;
    W1=W1+eta*Jw1;
end

yp=[0 0 zeros(1,length(k))];                  
yphat=[0 0 zeros(1,length(k))];
ei=zeros(1,length(k));
for i=3:length(k)+2
    u=sin(2*pi*(i-3)/25);
    yp(i)=f(yp(i-1),yp(i-2))+u;
    
    A1=[1 yp(i-1) yp(i-2)]*W1;
    y1=tanh(A1);
    A2=[1 y1]*W2;
    y2=tanh(A2);
    A3=[1 y2]*W3;
    N=A3;
    
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
save('bpa2_ex2_50','allrmse','allVAF','allconvtime');
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