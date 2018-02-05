function L = graphCons_Hy(V, kk)

Row = V;
Volume = V;
Part = 20;
X = size(Row,1)/Part;

All_Matrix = [];

Dv = zeros(1,size(Row,1));

%disp('Start to constract sub_X');
tic
for i = 0:Part-1
    Temp = Row(floor(i*X)+1:floor((i+1)*X), :);
    %if mod(i,5) == 0
    %    disp(i);
    %end
    Z = sqrt(Temp.^2*ones(size(Volume'))+ones(size(Temp))*(Volume').^2-2*Temp*Volume');
    Z = -Z;
    Z = 1./(1+exp(-Z));
    Kn = zeros(size(Volume,1),size(Temp,1));
    Kn = sparse(Kn);
    for k = 1:size(Kn,2)
        [T,indx]=sort(Z(k,:),'descend');
        ind=indx(1,2:kk+1);
        Dv(ind)  = Dv(ind) + 1;
        Kn(ind,k) = 1;     
    end
    
    All_Matrix = [All_Matrix Kn];
    clear Z Temp;
end
toc


%disp('Start to compute :');
tic
Dv = Dv + 1;
Dv = sparse(diag(sparse(Dv)));
De = ones(1,size(Row,1)) * 100;
De = sparse(diag(sparse(De)));
I = sparse(diag(sparse(ones(1,size(Row,1)))));
Kn = All_Matrix;
clear All_Matrix;

Ans = inv(Dv);
L = I - Ans * Kn * (De\Kn') * Ans;
L(isnan(L)) = 0;
toc
clear K;