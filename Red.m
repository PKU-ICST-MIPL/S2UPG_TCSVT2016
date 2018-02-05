function MatrixRed = Red(Matrix,Num1,Num2)
% *************************************************************************
% *************************************************************************
% Parameters:
% Matrix: similarity matrix for two set of patches 
% Num1: patch number for each media instance in 1st patch set 
% Num2: patch number for each media instance in 2nd patch set
%
% Here ia an example: 
% We have 2 images and 3 text. Each image has 2 patches and each text has 3
% patches.
% So we have a Matrix of (4*9), along with Num1 = [2, 2] and Num2 = [3,3,3].
% *************************************************************************
% *************************************************************************

MatrixRed = zeros(size(Num1,2),size(Num2,2));
IndexX = zeros(1,size(Num1,2)+1);
IndexY = zeros(1,size(Num2,2)+1);

IndexX(1) = 1;
IndexY(1) = 1;
for i = 1:size(Num1,2)
    IndexX(i+1) = IndexX(i) + Num1(i);
end

for i = 1:size(Num2,2)
    IndexY(i+1) = IndexY(i) + Num2(i);
end


for i = 1:size(Num1,2)
    for j = 1:size(Num2,2)
         MatrixRed(i,j) = mean(mean(Matrix( IndexX(i):IndexX(i+1)-1 , IndexY(j):IndexY(j+1)-1 ) )) ;
    end
end


end