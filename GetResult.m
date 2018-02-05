function [mapIA, mapIT, mapTA, mapTI] = GetResult(P_img, P_txt, I_tr_P, I_te_P, T_tr_P, T_te_P, trImgCat, trTxtCat, Num, I_tr_NP, I_te_NP, T_tr_NP, T_te_NP, trCatAll, teCatAll)

% *************************************************************************
% *************************************************************************
% Parameters:
% P_img: joint feature representation projection matrix of image
%              dimension : (d_c * d_I)
% P_txt: joint feature representation projection matrix of text
%              dimension : (d_c * d_T)
% I_tr_P: joint feature representation of image patches in training set
%              dimension : (I_tr_num_P * d_c)
% I_te_P: joint feature representation of image patches in test set
%              dimension : (I_te_num_P * d_c)
% T_tr_P: joint feature representation of text patches in training set
%              dimension : (T_tr_num_P * d_c)
% T_te_P: joint feature representation of texts patches in test set
%              dimension : (T_te_num_P* d_c)
% trImgCat: the category list of image patches in training set
%              dimension : I_tr_num_P * 1
% trTxtCat: the category list of text patches in training set
%              dimension : T_tr_num_P * 1
%
% I_tr_NP: joint feature representation of origianl images in training set
%              dimension : (I_tr_num_NP * d_c)
% I_te_NP: joint feature representation of origianl images in test set
%              dimension : (I_te_num_NP * d_c)
% T_tr_NP: joint feature representation of origianl texts in training set
%              dimension : (T_tr_num_NP * d_c)
% T_te_NP: joint feature representation of origianl texts in test set
%              dimension : (T_te_num_NP * d_c)
% trCatAll: the category list of original texts and images in training set
%              dimension : I_te_num_NP * 1
% teCatAll: the category list of original texts and images in test set
%              dimension : I_te_num_NP* 1
% Num: patch number for each media instance of two media types in test set 
%              dimension : I_te_num_P * 2 (1st dimenstion for image and 2nd dimension for text)
% *************************************************************************
% *************************************************************************

m = 1/3;
I_tr_P = I_tr_P.^m;
I_te_P = I_te_P.^m;
T_tr_P = T_tr_P.^m;
T_te_P = T_te_P.^m;   
I_tr_NP = I_tr_NP.^m;
I_te_NP = I_te_NP.^m;
T_tr_NP = T_tr_NP.^m;
T_te_NP = T_te_NP.^m;   

[I_tr_n_P I_te_n_P] = znorm(I_tr_P,I_te_P);
[T_tr_n_P T_te_n_P] = znorm(T_tr_P,T_te_P);
[I_tr_n_NP I_te_n_NP] = znorm(I_tr_NP,I_te_NP);
[T_tr_n_NP T_te_n_NP] = znorm(T_tr_NP,T_te_NP);

I_tr_red_P = (P_img' * I_tr_n_P')';
T_tr_red_P = (P_txt' * T_tr_n_P')';
I_te_red_P = (P_img' * I_te_n_P')';
T_te_red_P = (P_txt' * T_te_n_P')';
I_tr_red_NP = (P_img' * I_tr_n_NP')';
T_tr_red_NP = (P_txt' * T_tr_n_NP')';
I_te_red_NP = (P_img' * I_te_n_NP')';
T_te_red_NP = (P_txt' * T_te_n_NP')';

I_tr_num_P = size(I_tr_P,1);
I_tr_num_NP = size(I_tr_NP,1);
I_te_num_P = size(I_te_P,1);
I_te_num_NP = size(I_te_NP,1);
T_tr_num_P = size(T_tr_P,1);
T_tr_num_NP = size(T_tr_NP,1);
T_te_num_P = size(T_te_P,1);
T_te_num_NP = size(T_te_NP,1);


I_tr_red = [I_tr_red_NP ; I_tr_red_P];
I_te_red = [I_te_red_NP ; I_te_red_P];
T_tr_red = [T_tr_red_NP ; T_tr_red_P];
T_te_red = [T_te_red_NP ; T_te_red_P];

tr_n_I = I_tr_num_P + I_tr_num_NP;
te_n_I = I_te_num_P + I_te_num_NP;
tr_n_T = T_tr_num_P + T_tr_num_NP;
te_n_T = T_te_num_P + T_te_num_NP;
trImgCat = [trCatAll;trImgCat];
trTxtCat = [trCatAll;trTxtCat];


W_II =  Similarity(I_tr_red, I_te_red, I_tr_red, I_te_red, tr_n_I, te_n_I, tr_n_I, te_n_I, I_tr_num_NP, I_te_num_NP, I_tr_num_NP, I_te_num_NP, trImgCat, trImgCat);
W_II_NP = W_II(1:I_te_num_NP, te_n_I +1:te_n_I+I_te_num_NP);
W_II_P = W_II(I_te_num_NP+1:te_n_I, te_n_I+I_te_num_NP+1:end);
W_II_P_Red = Red(W_II_P, Num(1,:), Num(1,:));

clear W_II;
clear W_II_P;

W_IT =  Similarity(I_tr_red, I_te_red, T_tr_red, T_te_red, tr_n_I, te_n_I, tr_n_T, te_n_T, I_tr_num_NP, I_te_num_NP, T_tr_num_NP, T_te_num_NP, trImgCat, trTxtCat);
W_IT_NP = W_IT(1:I_te_num_NP, te_n_I +1:te_n_I+T_te_num_NP);
W_TI_NP = W_IT_NP';
W_IT_P = W_IT(I_te_num_NP+1:te_n_I, te_n_I+T_te_num_NP+1:end);
W_TI_P = W_IT_P';
W_IT_P_Red = Red(W_IT_P, Num(1,:), [Num(2,:)]);
clear W_IT;
clear W_IT_P;


W_TT =  Similarity(T_tr_red, T_te_red, T_tr_red, T_te_red, tr_n_T, te_n_T,tr_n_T, te_n_T, T_tr_num_NP, T_te_num_NP, T_tr_num_NP, T_te_num_NP,trTxtCat, trTxtCat);
W_TT_NP = W_TT(1:T_te_num_NP, te_n_T +1:te_n_T+T_te_num_NP);
W_TT_P = W_TT(T_te_num_NP+1:te_n_T, te_n_T+T_te_num_NP+1:end);
W_TT_P_Red = Red(W_TT_P, Num(2,:), Num(2,:));
clear W_TT;
clear W_TT_P;

W_NP = [W_II_NP,W_IT_NP;...
    W_IT_NP',W_TT_NP];

W_P = [W_II_P_Red,W_IT_P_Red;...
    W_IT_P_Red',W_TT_P_Red];

TestWI = 5*W_NP(1:I_te_num_NP,:) +  W_P(1:I_te_num_NP,:);
TestWT = W_NP(I_te_num_NP+1:end,:) +  W_P(I_te_num_NP+1:end,:);
TestWIT = 5*W_NP(1:I_te_num_NP,1+I_te_num_NP:end)+  W_P(1:I_te_num_NP,1+I_te_num_NP:end);
TestWTI = W_NP(1+I_te_num_NP:end,1:I_te_num_NP)+  W_P(1+I_te_num_NP:end,1:I_te_num_NP);

disp(['image query all... ']);
[ mapIA, A, B] = evaluateMAPPR(TestWI, teCatAll, [teCatAll;teCatAll]);
disp(['text query all... ']);
[ mapTA, A, B] = evaluateMAPPR(TestWT, teCatAll, [teCatAll;teCatAll]);
disp(['image query text... ']);
[ mapIT, A, B] = evaluateMAPPR(TestWIT, teCatAll, teCatAll);
disp(['text query image... ']);
[ mapTI, A, B] = evaluateMAPPR(TestWTI, teCatAll, teCatAll);

