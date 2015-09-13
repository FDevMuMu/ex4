function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%%%%%%%%%%%%%%%
% [1] Forward propagation
X = [ones(m, 1) X]; % add bias
a1 = X;
a2 = sigmoid(X*Theta1');  % z1 = X*Theta1'
a2 = [ones(m,1), a2];     % add bias 
a3 = sigmoid(a2*Theta2'); % z2 = a2*Theta2', h_theta = a3
h_theta = a3;
yk = zeros(num_labels, m); % every colunm is the y of one example, m is the number of examples
for i=1:m,
  yk(y(i),i) = 1;       % m vectors in 0 and 1 of every exmaple

end
%cost function
J = (1/m)*sum( sum ( (-yk).*log(h_theta') - (1-yk).*log(1 - h_theta')));

%regularization
     % note that you shouldn't regularizing the theta0(the bias)

regtheta1 = Theta1(:,2:size(Theta1,2));
regtheta2 = Theta2(:,2:size(Theta2,2));

Reg = lambda * ( sum( sum(regtheta1.^2)) + sum( sum( regtheta2.^2))) / (2*m);

J = J + Reg;

%  [2] Backpropagation

for t=1:m,

  % (1) forward propagation
  a1 = X(t,:)';
  %z2 = Theta1*a1'; % Theta is trans of theta

  z2 = Theta1*a1;
  a2 = sigmoid(z2);
  a2 = [1; a2]; %add bias

  %z3 = Theta2*a2;
  z3 = Theta2*a2;

  a3 = sigmoid(z3); % the last layer, h_theta is equal to a3

  % (2)  Back propagation
  %z2=[1; z2]; % bias
  delta3 = a3 - yk(:,t);
  %delta2 = (Theta2' * delta3).*sigmoidGradient(z2);
  delta2 = (Theta2' * delta3).*(a2.*(1-a2));
  % remove the sigma2(0)

  delta2 = delta2(2:end);

  Theta2_grad = Theta2_grad + delta3 * a2';
  Theta1_grad = Theta1_grad + delta2 * a1';

end

    Theta1_grad(:, 1) = Theta1_grad(:, 1)./m;
    Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end)./m + ((lambda/m) * Theta1(:, 2:end));
  
  
  Theta2_grad(:, 1) = Theta2_grad(:, 1)./m;
  
  Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end)./m + ((lambda/m) * Theta2(:, 2:end));    


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
