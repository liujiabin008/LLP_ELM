function [f,R_pos,p_all] = Wilcoxon(A,alpha)
% alpha: significant level
m=size(A,2);
Sig=zeros(m,m);
R_pos = zeros(m,m);
p_all = zeros(m,m);
for i=m
    for j=1:m
        x=A(:,i);
        y=A(:,j);
        [p,h,stats]=signrank(x,y,'alpha',alpha);
        R_pos(i,j) = stats.signedrank;
        p_all(i,j) = p;
%          h=1;
        if mean(x)>mean(y) && h==1
            Sig(i,j)=1;
        elseif mean(x)<mean(y) && h==1
            Sig(i,j)=-1;
        else
            Sig(i,j)=0;
        end
    end
end

X=Sig-Sig';
f=[sum(X==1,2) sum(X==0,2)-1 sum(X==-1,2)];

end