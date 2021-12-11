train_T = readtable('data/train_data.csv','ReadVariableNames',false);
test_T = readtable('data/test_data.csv','ReadVariableNames',false);

X_train = train_T{:,1:2};
y_train = train_T{:,3};
n_train = length(y_train);


X_test = test_T{:,1:2};
y_test = test_T{:,3};
n_test = length(y_test);
Nt = X_test((y_test == -1),:);
Pt = X_test((y_test == 1),:);

d = 2;
C = 1;
Cf = [0.1, 0.3, 0.5, 0.7, 0.9];
cf_len = length(Cf);
f_attack = [0, 0.3, 0.5, 0.7, 1.0];
fa_len = length(f_attack);

output_adsvm = zeros(cf_len, fa_len);
output_wtrain = zeros(cf_len, d);
output_btrain = zeros(cf_len, 1);

maxx = zeros(1,d);
minn = zeros(1,d);
for k = 1:d
    maxx(1,k) = max(X_train(:,k));
    minn(1,k) = min(X_train(:,k));
end

X_max = zeros(n_train,d);
X_min = zeros(n_train,d);

for k = 1:d
    X_max(:, k) = maxx(1,k) - X_train(:,k);
    X_min(:, k) = minn(1,k) - X_train(:,k);
end

ColOfOnes = ones(d,1);

for i=1:cf_len
    cf = Cf(i);
    cvx_begin quiet                       
        variable w(d) 
        variable b;
        variable xi(n_train);
        variable t(n_train);
        variable u(n_train,d);
        variable v(n_train,d);

        minimize 1/2 *(norm(w)) + C*sum(xi);
        subject to
            xi >= 0;
            xi - 1 + y_train.*(X_train*w+b)- t >= 0;
            t >= cf*(((v.*X_max) - (u.*X_min))*ColOfOnes);
            u-v == 0.5*repmat((1+y_train),1,d).*repmat(w',n_train,1);
            u >= 0;
            v >= 0;    
    cvx_end
    output_wtrain(i,:) = w;
    output_btrain(i,1) = b;
    
    for j=1:fa_len
        correct = 0;
        fa = f_attack(j);
        Ntr = reshape(Nt(randperm(length(Nt)*2)), length(Nt), 2);
        Ptattk = Pt+fa*(Ntr-Pt);
        X_test_attk = [Ptattk;Nt];
        
        ypred = Nt*w+b;
        correct = correct+sum(ypred<=0);
        
        ypred = Ptattk*w+b;
        correct = correct + sum(ypred>0);
        output_adsvm(i,j) = (correct/length(X_test_attk))*100;
    end
    fprintf('cf = %f done\n', cf);  
end