function [P_img, P_txt] = learnProjection(I_tr, I_te, T_tr, T_te, pair, img_Dim, txt_Dim, tr_n_I, tr_n_T, L_cat_img, L_cat_txt, cat_Num, param, k)

iteration = param.iteration;
gamma = param.gamma;
sigma = param.sigma;
miu = param.miu;

[I_tr_n I_te_n] = znorm(I_tr,I_te);
[T_tr_n T_te_n] = znorm(T_tr,T_te);

rand('seed', 1);
P_img = rand(img_Dim, cat_Num);
rand('seed', 1);
P_txt = rand(txt_Dim, cat_Num);
count = 1;

lastLoss = 0;

P_img_back = P_img;
P_txt_back = P_txt;

for i = 1:iteration

    P_img_back = P_img;
    P_txt_back = P_txt;
    
    I_proj = P_img' * [I_tr_n;I_te_n]';
    T_proj = P_txt' * [T_tr_n;T_te_n]';

    LA = graphCons_Hy([I_proj,T_proj]', k);
    
    O_O = [I_proj, T_proj]';

    b_img = L_cat_img(1:tr_n_I,:) - I_tr_n * P_img;
    b_img = ones(size(b_img, 1)) * b_img;
    b_img = (1/size(b_img, 1)) * b_img;
    
    b_txt = L_cat_txt(1:tr_n_T,:) - T_tr_n * P_txt;
    b_txt = ones(size(b_txt, 1)) * b_txt;
    b_txt = (1/size(b_txt, 1)) * b_txt;
    
    lossValue = sigma * trace(O_O' * LA * O_O) + ...
        gamma*norm21(P_img) + gamma*norm21(P_txt) + ...
        miu*sum(sum((I_tr_n * P_img - L_cat_img(1:tr_n_I,:) + b_img).^2)) + miu*sum(sum((T_tr_n * P_txt - L_cat_txt(1:tr_n_T,:) + b_txt).^2));

    disp(['iteration ' num2str(i) ': ' num2str(lossValue) ', change ratio:' num2str((lastLoss - lossValue) / lastLoss)]);
    
    I_a = [I_tr_n;I_te_n];
    T_a = [T_tr_n;T_te_n];

    I_tr_red = (P_img' * I_tr_n')';
    T_tr_red = (P_txt' * T_tr_n')';
    I_te_red = (P_img' * I_te_n')';
    T_te_red = (P_txt' * T_te_n')';

    [N_img, a] = size(L_cat_img);
    [N_txt, b] = size(L_cat_txt);
    LX = LA(1:N_img, 1:N_img);
    LY = LA(N_img+1:N_img+N_txt, N_img+1:N_img+N_txt);
    LXY = LA(1:N_img, N_img+1:N_img+N_txt);
    LYX = LXY';
    D1 = zeros(img_Dim, img_Dim);
    D2 = zeros(txt_Dim, txt_Dim);
    for D1_i = 1:img_Dim
        D1(D1_i,D1_i) = 1/(2*norm(P_img(D1_i, :)));
    end
    for D2_i = 1:txt_Dim
        D2(D2_i,D2_i) = 1/(2*norm(P_txt(D2_i, :)));
    end
    H_1 = eye(tr_n_I) - (1/tr_n_I);
    H_2 = eye(tr_n_T) - (1/tr_n_T);   
    
    for j = 1:1
        P_img_next = inv(miu*I_tr_n'*H_1*H_1*I_tr_n + gamma*D1 + sigma*I_a'*LX*I_a) * (miu*I_tr_n'*H_1*L_cat_img(1:tr_n_I,:) - sigma*I_a'*LXY*T_a*P_txt);
        P_txt_next = inv(miu*T_tr_n'*H_2*H_2*T_tr_n + gamma*D2 + sigma*T_a'*LY*T_a) * (miu*T_tr_n'*H_2*L_cat_txt(1:tr_n_T,:) - sigma*T_a'*LYX*I_a*P_img);
        P_img = P_img_next;
        P_txt = P_txt_next;
    end

    lastLoss = lossValue;

end

P_img = P_img_back;
P_txt = P_txt_back;