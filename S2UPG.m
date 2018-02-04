function [mapIA, mapIT, mapTA, mapTI] = S2UPG(I_tr_NP, T_tr_NP, I_te_NP, T_te_NP, ...
    I_tr_P, T_tr_P, I_te_P, T_te_P, trainCat, testCat, gamma, sigma, miu, k)

% *************************************************************************
% *************************************************************************
% Parameters:
% I_tr_NP: the feature matrix of image instances for training
%              dimension : tr_n * d_i
% T_tr_NP: the feature matrix of text instances for training
%              dimension : tr_n * d_t
% I_te_NP: the feature matrix of image instances for test
%              dimension : te_n * d_i
% T_te_NP: the feature matrix of text instances for test
%              dimension : te_n * d_t
% I_tr_P: the feature matrix of image patches for training
%              dimension : tr_n * d_i
% T_tr_P: the feature matrix of text patches for training
%              dimension : tr_n * d_t
% I_te_P: the feature matrix of image patches for test
%              dimension : te_n * d_i
% T_te_P: the feature matrix of text patches for test
%              dimension : te_n * d_t
% trainCat: the category list of data for training
%              dimension : tr_n * 1
% testCat: the category list of data for test
%              dimension : te_n * 1
% gamma: sparse regularization parameter, default: 1000
% sigma: mapping regularization parameter, default: 0.1
% miu: high level regularization parameter, default: 10
% k: kNN parameter, default: 100
% *************************************************************************
% *************************************************************************

warning off;

T_tr_P = T_tr_P./repmat(sum(T_tr_P, 2), 1, 10);
T_te_P = T_te_P./repmat(sum(T_te_P, 2), 1, 10);

I_tr_NP = (I_tr_NP + I_tr_P)/2;
I_te_NP = (I_te_NP + I_te_P)/2;
T_tr_NP = (T_tr_NP + T_tr_P)/2;
T_te_NP = (T_te_NP + T_te_P)/2;

I_tr_NP = I_tr_NP.^1/3;
I_te_NP = I_te_NP.^1/3;
T_tr_NP = T_tr_NP.^1/3;
T_te_NP = T_te_NP.^1/3;

trImgCat_NP = trainCat;
trTxtCat_NP = trainCat;

img_Dim = size(I_tr_NP, 2);
txt_Dim = size(T_tr_NP, 2);
cat_Num = max(trainCat);

tr_n_I_NP = size(I_tr_NP, 1);
te_n_I_NP = size(I_te_NP, 1);
tr_n_T_NP = size(T_tr_NP, 1);
te_n_T_NP = size(T_te_NP, 1);
tr_n_I_P = size(I_tr_P, 1);
te_n_I_P = size(I_te_P, 1);
tr_n_T_P = size(T_tr_P, 1);
te_n_T_P = size(T_te_P, 1);
L_cat_img_NP = zeros(tr_n_I_NP + te_n_I_NP, cat_Num);
L_cat_txt_NP = zeros(tr_n_T_NP + te_n_T_NP, cat_Num);
L_cat_img_P = zeros(tr_n_I_P + te_n_I_P, cat_Num);
L_cat_txt_P = zeros(tr_n_T_P + te_n_T_P, cat_Num);

for i = 1:tr_n_I_NP
    L_cat_img_NP(i,trImgCat_NP(i)) = 1;
end
for i = 1:tr_n_T_NP
    L_cat_txt_NP(i,trTxtCat_NP(i)) = 1;
end
for i = 1:tr_n_I_P
    L_cat_img_P(i,trImgCat_NP(i)) = 1;
end
for i = 1:tr_n_T_P
    L_cat_txt_P(i,trTxtCat_NP(i)) = 1;
end

if tr_n_I_P < tr_n_T_P
    pair = zeros(tr_n_I_P,2);
    for i = 1:tr_n_I_P
        IndexSet = find(trTxtCat_P == trImgCat_P(i));
        Index = IndexSet(randint(1,1,(size(IndexSet,1)))  +1 );
        pair(i,1) = i;
        pair(i,2) = Index;
    end
else
    pair = zeros(tr_n_T_P,2);
    for i = 1:tr_n_T_P
        IndexSet = find(trImgCat_NP == trTxtCat_NP(i));
        Index = IndexSet(randint(1,1,(size(IndexSet,1)))  +1 );
        pair(i,1) = Index;
        pair(i,2) = i;
    end     
end

param.iteration = 3;
param.gamma = gamma;
param.sigma = sigma;
param.miu = miu;
[P_img, P_txt] = learnProjection(I_tr_NP, I_te_NP, T_tr_NP, T_te_NP, pair, img_Dim, txt_Dim, tr_n_I_NP, tr_n_T_NP, L_cat_img_NP, L_cat_txt_NP, cat_Num, param);

[I_tr_n, I_te_n] = znorm(I_tr_NP,I_te_NP);
[T_tr_n, T_te_n] = znorm(T_tr_NP,T_te_NP);
I_tr_red = (P_img' * I_tr_n')';
T_tr_red = (P_txt' * T_tr_n')';
I_te_red = (P_img' * I_te_n')';
T_te_red = (P_txt' * T_te_n')'; 

W_II = unifyKnnKernel(I_tr_red, I_te_red, I_tr_red, I_te_red, tr_n_I_NP, te_n_I_NP, tr_n_I_NP, te_n_I_NP, trImgCat_NP, trImgCat_NP, k);
W_IT = unifyKnnKernel(I_tr_red, I_te_red, T_tr_red, T_te_red, tr_n_I_NP, te_n_I_NP, tr_n_T_NP, te_n_T_NP, trImgCat_NP, trTxtCat_NP, k);
W_TT = unifyKnnKernel(T_tr_red, T_te_red, T_tr_red, T_te_red, tr_n_T_NP, te_n_T_NP, tr_n_T_NP, te_n_T_NP, trTxtCat_NP, trTxtCat_NP, k);

W_II = W_II(1:te_n_I_NP, 1:te_n_I_NP);
W_TT = W_TT(1:te_n_T_NP, 1:te_n_T_NP);
W_IT = W_IT(1:te_n_I_NP, te_n_I_NP+1:te_n_I_NP+te_n_T_NP);
W = [W_II,W_IT;...
    W_IT',W_TT];

for i = 1:length(W)
    W(i,i) = 9999;
end

WI = W(1:te_n_I_NP,:);
WT = W(te_n_I_NP+1:te_n_I_NP+te_n_T_NP,:);

disp(['image query all... ']);
[ mapIA, ~, ~] = evaluateMAPPR(WI, testCat, [testCat;testCat]);
disp(['text query all... ']);
[ mapTA, ~, ~] = evaluateMAPPR(WT, testCat, [testCat;testCat]);
disp(['image query text... ']);
[ mapIT, ~, ~] = evaluateMAPPR(W_IT, testCat, testCat);
disp(['text query image... ']);
[ mapTI, ~, ~] = evaluateMAPPR(W_IT', testCat, testCat);