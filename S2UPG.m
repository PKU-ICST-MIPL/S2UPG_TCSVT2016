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
% 
% For convenience, we let I_tr_P be the average vector of all patches
% for every origianl training image, and this goes for I_te_P, T_tr_P and T_te_P.
% *************************************************************************
% *************************************************************************

warning off;

T_tr_P = T_tr_P./repmat(sum(T_tr_P, 2), 1, 10);
T_te_P = T_te_P./repmat(sum(T_te_P, 2), 1, 10);

I_tr = (I_tr_NP + I_tr_P)/2;
I_te = (I_te_NP + I_te_P)/2;
T_tr = (T_tr_NP + T_tr_P)/2;
T_te = (T_te_NP + T_te_P)/2;

I_tr = I_tr.^1/3;
I_te = I_te.^1/3;
T_tr = T_tr.^1/3;
T_te = T_te.^1/3;

trImgCat = trainCat;
trTxtCat = trainCat;

img_Dim = size(I_tr, 2);
txt_Dim = size(T_tr, 2);
cat_Num = max(trainCat);

tr_n_I = size(I_tr, 1);
te_n_I = size(I_te, 1);
tr_n_T = size(T_tr, 1);
te_n_T = size(T_te, 1);
tr_n_I_P = size(I_tr_P, 1);
te_n_I_P = size(I_te_P, 1);
tr_n_T_P = size(T_tr_P, 1);
te_n_T_P = size(T_te_P, 1);
L_cat_img = zeros(tr_n_I + te_n_I, cat_Num);
L_cat_txt = zeros(tr_n_T + te_n_T, cat_Num);
L_cat_img_P = zeros(tr_n_I_P + te_n_I_P, cat_Num);
L_cat_txt_P = zeros(tr_n_T_P + te_n_T_P, cat_Num);

for i = 1:tr_n_I
    L_cat_img(i,trImgCat(i)) = 1;
end
for i = 1:tr_n_T
    L_cat_txt(i,trTxtCat(i)) = 1;
end
for i = 1:tr_n_I_P
    L_cat_img_P(i,trImgCat(i)) = 1;
end
for i = 1:tr_n_T_P
    L_cat_txt_P(i,trTxtCat(i)) = 1;
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
        IndexSet = find(trImgCat == trTxtCat(i));
        Index = IndexSet(randint(1,1,(size(IndexSet,1)))  +1 );
        pair(i,1) = Index;
        pair(i,2) = i;
    end     
end

param.iteration = 3;
param.gamma = gamma;
param.sigma = sigma;
param.miu = miu;
[P_img, P_txt] = learnProjection(I_tr, I_te, T_tr, T_te, pair, img_Dim, txt_Dim, tr_n_I, tr_n_T, L_cat_img, L_cat_txt, cat_Num, param, k);

save('PIT_Hy', 'P_img', 'P_txt');

Num = ones(2,te_n_I);  %For convenience, the patch number of each media instance in this example code is set to be 1. See detatils in Similarity.m and Red.m

[mapIA, mapIT, mapTA, mapTI] = GetResult(P_img, P_txt, I_tr_P, I_te_P, T_tr_P, T_te_P, trainCat, trainCat, Num, I_tr_NP, I_te_NP, T_tr_NP, T_te_NP, trainCat, testCat);

