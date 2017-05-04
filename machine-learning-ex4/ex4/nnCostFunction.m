function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

cal_type = 1; % 1 Matrix 2 For-Loop
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
if (cal_type == 1)
	Y = zeros(m,num_labels);
	for i=1:m
		%Y(i,rem(y(i),num_labels)+1)=1; 不用把10转成的0放在第一位
		Y(i,y(i))=1;
	endfor
	%%Theta1: hidden_layer_size * (input_layer_size + 1)
	%%Theta2: num_labels * (hidden_layer_size + 1)

	% 使用矩阵进行计算  calculate by using matix
	a1 = [ones(m,1) X]; 		%m*(input_layer_size+1)
	z2 = a1*Theta1'; 		%m*hidden_layer_size
	a2 = [ones(m,1) sigmoid(z2)]; 	%m*(hidden_layer_size+1)
	z3 = a2*Theta2'; 		%m*num_labels
	h  = sigmoid(z3);		%m*num_labels
	%J = (-Y.*log(h)-(1-Y).*log(1-h));
	J = 1/m*sum(sum((-Y.*log(h)-(1-Y).*log(1-h))));
elseif (cal_type == 2)
% calculate  using for-loop
	for i=1:m
		fprintf('Calculating X(%i)...\r',i)
		a1=[1 X(i,:)]; %1*(input_layer_size+1) row vector
		z2 = a1*Theta1';
		a2 = [1 sigmoid(z2)];
		z3 = a2*Theta2';
		h  = sigmoid(z3);
		yy = zeros(1,num_labels);
		yy(y(i)) = 1;
		for k=1:num_labels
			J = J + (-yy(k)*log(h(k))-(1-yy(k))*log(1-h(k)));
		endfor
		if exist('OCTAVE_VERSION')
			fflush(stdout);
		endif
	endfor
	%Do not forget to devide J by m
	J = J/m;
endif
%% Regularized cost function
%%Note that you should not be regularizing the terms that correspond to the bias.
J = J + lambda/(2*m)*(sum(sum(Theta1(:,2:end).*Theta1(:,2:end)))+sum(sum(Theta2(:,2:end).*Theta2(:,2:end))));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

if (cal_type == 1)

	Y = zeros(m,num_labels);
	for i=1:m
		%Y(i,rem(y(i),num_labels)+1)=1; 不用把10转成的0放在第一位
		Y(i,y(i))=1;
	endfor
	%%Theta1: hidden_layer_size * (input_layer_size + 1)
	%%Theta2: num_labels * (hidden_layer_size + 1)

	% 使用矩阵进行计算  calculate by using matix
	a1 = [ones(m,1) X]; 		%m*(input_layer_size+1)
	z2 = a1*Theta1'; 		%m*hidden_layer_size
	a2 = [ones(m,1) sigmoid(z2)]; 	%m*(hidden_layer_size+1)
	z3 = a2*Theta2'; 		%m*num_labels
	a3 = sigmoid(z3);		%m*num_labels
	Delta_1 = zeros(size(Theta1));
	Delta_2 = zeros(size(Theta2));
	delta_3 = a3 - Y; %m*num_labels

	Delta_2 = Delta_2 + delta_3'*a2;
	% Note to remove Theta2(:,1)
	delta_2 = delta_3*Theta2(:,2:end).*sigmoidGradient(z2); %m*(hidden_layer_size)
	%delta_2 = delta_2(:,2:end); %m*hidden_layer_size
	%delta_1 = delta_2*Theta1.*sigmoidGradient(a1); %m*(input_layer_size+1)
	Delta_1 = Delta_1 + delta_2'*a1; % hidden_layer_size * (input_layer_size+1)
	Theta1_grad = Delta_1/m;
	Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);
	Theta2_grad = Delta_2/m;
	Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);
elseif (cal_type == 2)
	Delta_1 = zeros(size(Theta1));
	Delta_2 = zeros(size(Theta2));
	
	for i=1:m
		fprintf('Applying Back Propagation on X(%i)...\r',i)
		a1=[1;X(i,:)']; % (input_layer_size+1) * 1 column vector
		z2 = Theta1*a1; % hidden_layer_size * 1
		a2 = [1;sigmoid(z2)]; %(hidden_layer_size+1)*1
		z3 = Theta2*a2; %num_labels * 1
		a3 = sigmoid(z3); %num_labels * 1
		yy = zeros(num_labels,1); %num_labels * 1
		yy(y(i)) = 1;
		delta_2 = zeros(hidden_layer_size,1);
		delta_3 = zeros(num_labels,1); %num_labels*1
		delta_3 = a3 - yy;
		
		%%Theta1: hidden_layer_size * (input_layer_size + 1)
		%%Theta2: num_labels * (hidden_layer_size + 1)
		delta_2 = Theta2(:,2:end)'*delta_3.*sigmoidGradient(a2); %(hidden_layer_size + 1) * 1
		%delta_2 = delta_2(2:end); %(hidden_layer_size) * 1
		Delta_2 = Delta_2 + delta_3*a2'; % num_labels*(hidden_layer_size+1)
		%delta_1 = Theta1'*delta_2.*sigmoidGradient(a1); %(input_layer_size + 1) * 1
		Delta_1 = Delta_1 + delta_2*a1';	% hidden_layer_size*(input_layer_size+1)
		if exist('OCTAVE_VERSION')
			fflush(stdout);
		endif
	endfor
	Theta1_grad = Delta_1/m;
	Theta2_grad = Delta_2/m;
	Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);
	Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);
	
endif



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end