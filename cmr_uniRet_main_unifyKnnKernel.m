function W_dot = cmr_uniRet_main_unifyKnnKernel(I_tr_red, I_te_red, T_tr_red, T_te_red,tr_n_I, te_n_I, tr_n_T, te_n_T, trImgCat, trTxtCat)
Part = 10;
Row = [I_te_red;T_te_red];
Volume = [I_tr_red;T_tr_red];
X = size(Row,1)/Part;

for i = 0:Part-1
    sub_X{i+1} = Row(round(i*X)+1:round(i*X + X), :);
end

Count = 0;

kk = 100;
for i = 1:Part
        Z = sqrt(sub_X{i}.^2*ones(size(Volume'))+ones(size(sub_X{i}))*(Volume').^2-2*sub_X{i}*Volume');
        Z = -Z;
        Z = 1./(1+exp(-Z));
        Kn{i}=zeros(size(sub_X{i},1),size(Volume,1));              
        for k = 1:size(sub_X{i},1)
            [Ki,indx]=sort(Z(k,:),'descend');
            if(~mod(k, 10000)) disp([' i ' num2str(i)  ' k ' num2str(k)]); end
            ind=indx(1:kk);
            Kn{i}(k,ind)=Z(k,ind);
        end
        disp(i);
        clear Z;  
        Kn{i} = sparse(Kn{i});
end


Sub_Knn = Kn{1};
for i = 2:Part
    Sub_Knn = [Sub_Knn;Kn{i}];
end


disp('Knn Done!');

WI = Sub_Knn(1:te_n_I, :);
WT = Sub_Knn(te_n_I+1:te_n_I+te_n_T, :);

WI_s = sum(WI, 2);
WT_s = sum(WT, 2);
WI = WI./repmat(WI_s, 1, tr_n_I+tr_n_T);
WT = WT./repmat(WT_s, 1, tr_n_T+tr_n_I);
Sub_Knn = [WI;WT];

Y0 = double((repmat([trImgCat;trTxtCat],1,(tr_n_I+tr_n_T)))==(repmat([trImgCat;trTxtCat],1,(tr_n_I+tr_n_T)))');

disp('Y0 Done!');

W_dot = Sub_Knn * Y0 * Sub_Knn';

end

