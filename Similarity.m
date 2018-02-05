function W_dot = Similarity(I_tr_red, I_te_red, T_tr_red, T_te_red,tr_n_I, te_n_I, tr_n_T, te_n_T, I_tr_num_NP, I_te_num_NP, T_tr_num_NP, T_te_num_NP, trImgCat, trTxtCat)
% *************************************************************************
% *************************************************************************
% Parameters:
% I_tr_red: joint feature representation of images (and their patches) in training set
%              dimension : (tr_n_I * d_c)
% I_te_red: joint feature representation of images (and their patches) in test set
%              dimension : (te_n_I * d_c)
% T_tr_red: joint feature representation of texts (and their patches) in training set
%              dimension : (tr_n_T * d_c)
% T_te_red: joint feature representation of texts (and their patches) in test set
%              dimension : (te_n_T * d_c)
% tr_n_I: the number of images (and their patches) in training set
% te_n_I: the number of images (and their patches) in test set
% tr_n_T: the number of texts (and their patches) in training set
% te_n_T: the number of texts (and their patches) in test set
% I_tr_num_NP: the number of original images in training set
% I_te_num_NP: the number of original images in test set
% T_tr_num_NP: the number of origianl texts in training set
% T_te_num_NP: the number of original texts in test set
% trImgCat: the category list of images (and their patches)  in training set
%              dimension : tr_n_I * 1
% trTxtCat: the category list of texts (and their patches)  in training set
%              dimension : te_n_T * 1
% *************************************************************************
% *************************************************************************

Part = 10;
Row = [I_te_red;T_te_red];
Volume = [I_tr_red;T_tr_red];
X = size(Row,1)/Part;

for i = 0:Part-2
    sub_X{i+1} = Row(round(i*X)+1:round(i*X + X), :);
end
    sub_X{Part} = Row(round((Part-1)*X)+1:end, :);

Count = 0;

kk = 100;
for i = 1:Part
        Z = sqrt(sub_X{i}.^2*ones(size(Volume'))+ones(size(sub_X{i}))*(Volume').^2-2*sub_X{i}*Volume');
        Z = -Z;
        Z = 1./(1+exp(-Z));
        Kn{i}=zeros(size(sub_X{i},1),size(Volume,1));      
        
        for k = 1:size(sub_X{i},1)
            Count = Count+1;
            if ( (Count <= I_te_num_NP) || (Count>te_n_I && Count <=te_n_I+ I_te_num_NP))
                Z(k,I_tr_num_NP+1:tr_n_I) = Z(k,I_tr_num_NP+1:tr_n_I) * 0.85;
                Z(k,T_tr_num_NP+tr_n_I+1:end) = Z(k,T_tr_num_NP+tr_n_I+1:end) * 0.85;
            else
                Z(k,1:I_tr_num_NP) = Z(k,1:I_tr_num_NP)*0.85;
                Z(k,tr_n_I+1:T_tr_num_NP+tr_n_I) = Z(k,tr_n_I+1:T_tr_num_NP+tr_n_I)*0.85;
            end
            [Ki,indx]=sort(Z(k,:),'descend');
            if(~mod(k, 10000)) disp([' i ' num2str(i)  ' k ' num2str(k)]); end
            ind=indx(1:kk);
            Kn{i}(k,ind)=Z(k,ind);
        end
        %disp(i);
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

