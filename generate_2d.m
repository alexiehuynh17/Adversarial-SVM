A = [-1, -2]; B = [1, 2];  
Sigma = [1 .5; 0.5 2]; R = chol(Sigma);
m = 150; 
n = 1;

P = normc(repmat(A,m,1) + randn(m,2)*R); % positive data
[rowp, ~] = size(P);
N = normc(repmat(B,m,1) + randn(m,2)*R); % negative data
[rown, ~] = size(N);
X_train = [P;N];

PL = ones(rowp,1);
NL = -ones(rown,1);
y_train = [PL;NL];

Xy_train =[X_train, y_train];

Pt = normc(repmat(A,m,1) + randn(m,2)*R);
[rowp, ~] = size(Pt);
Nt = normc(repmat(B,m,1) + randn(m,2)*R);
[rown, ~] = size(Nt);
X_test = [Pt;Nt];
y_test = ones(rowp,1);
y_test(rowp+1:rowp+rown) = -1;

Xy_test = [X_test, y_test];

T_train = array2table(Xy_train);
T_test = array2table(Xy_test);
T_train.Properties.VariableNames(1:3) = {'x1','x2','label'};
T_test.Properties.VariableNames(1:3) = {'x1','x2','label'};
writetable(T_train,'data/train_data.csv')
writetable(T_test,'data/test_data.csv')