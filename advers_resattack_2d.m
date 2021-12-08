train_T = readtable('data/train_data.csv','ReadVariableNames',false);
test_T = readtable('data/test_data.csv','ReadVariableNames',false);

X_train = train_T{:,1:2};
y_train = train_T{:,3};
n_train = length(y_train);
N = X_train((y_train == -1),:);
P = X_train((y_train == 1),:);

X_test = test_T{:,1:2};
y_test = test_T{:,3};
n_test = length(y_test);
Nt = X_test((y_test == -1),:);
Pt = X_test((y_test == 1),:);

d = 2;
C = 1;
cdf = 1;

Cd = [0.9, 0.7, 0.5, 0.3, 0.1];
cd_len = length(Cd);

f_attack = [0, 0.3, 0.5, 0.7, 1.0];
fa_len = length(f_attack);

output_adsvm = zeros(cd_len, fa_len);

x_mean = mean(N);                        
x_t = repmat(x_mean,length(P),1);
ColOfOnes = ones(d,1);

for i=1:cd_len
    cd = Cd(i);
    
    e = cdf*((1- cd*abs(x_t - P)./(abs(x_t)+abs(P))).*((x_t-P).^2));
    ze = zeros(length(N),d);
    e = [e; ze];
    xe = x_t-P;
    xe = [xe;ze];
    
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
            t - (u.*e)*ColOfOnes >= 0;
            (v - u).*xe - 0.5*repmat((1+y_train),1,d).*repmat(w',n_train,1) == 0;
            u >= 0;
            v >= 0;       
    cvx_end
    
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
        output_adsvm(i,j) = (correct/n_test)*100;
    end
end