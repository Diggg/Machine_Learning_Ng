function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%
%You should write any additional code necessary
%to help you search over the parameters C and $. For both C and $, we
%suggest trying values in multiplicative steps (e.g., 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30).
%Note that you should try all possible pairs of values for C and $ (e.g., C = 0.3
%and $ = 0.1). For example, if you try each of the 8 values listed above for C
%and for $2, you would end up training and evaluating (on the cross validation
%set) a total of 82 = 64 different models.

C_list = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_list = [0.01 0.03 0.1 0.3 1 3 10 30];
m = size(C_list,2);
n = size(sigma_list,2);
errors = zeros(m,n);
for i = 1:m
	for j = 1:n
		C = C_list(i);
		sigma = sigma_list(j);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
		predictions = svmPredict(model, Xval);
		error = mean(double(predictions ~= yval));
		errors(i,j) = error;
		%fprintf('\nError=%f when C=%f sigma=%f.',error,C,sigma);
	end
end
% 找到最小error的C 和sigma
ind = find(errors == min(errors(:)));
[x,y] = ind2sub([m n],ind);

C = C_list(x);
sigma = sigma_list(y);
fprintf('\nGot Min Error=%f when C=%f sigma=%f.',error,C,sigma);
% =========================================================================

end
