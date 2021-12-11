train_T = readtable('data/spam_train.csv','ReadVariableNames',false);
test_T = readtable('data/spam_test.csv','ReadVariableNames',false);

%train_T = readtable('data/train_data.csv','ReadVariableNames',false);
%test_T = readtable('data/test_data.csv','ReadVariableNames',false);

d = length(train_T.Properties.VariableNames);
X_train = train_T{:,1:d-1};
y_train = train_T{:,d};
n_train = length(y_train);

X_test = test_T{:,1:d-1};
y_test = test_T{:,d};
n_test = length(y_test);

Nt = X_test((y_test == -1),:);
Pt = X_test((y_test == 1),:);

C = 1;
cdf = 1;

Cd = [0.9, 0.7, 0.5, 0.3, 0.1];
%Cd = [0.9];
cd_len = length(Cd);

f_attack = [0, 0.3, 0.5, 0.7, 1.0];
%f_attack = [0.3];
fa_len = length(f_attack);

output_adsvm = zeros(cd_len, fa_len);
output_svm = zeros(cd_len, fa_len);

round = 10;

for i=1:cd_len
    cd = Cd(i);  
    for j=1:fa_len
        correct = 0;
        svmcorrect = 0;
        fa = f_attack(j);
        fprintf('fa = %f\n', fa);
        for r=1:round
            fprintf('round = %d\n', r);
            non_NaN = 0;
            while non_NaN == 0
                rows = randperm(n_train, 0.1*n_train); 
                X_build = X_train(rows, :);
                y_build = y_train(rows, 1);

                N = X_build((y_build == -1),:);
                P = X_build((y_build == 1),:);
                PN = [P; N];
                PNL = [ones(length(P),1); -1*ones(length(N),1)];
                n_build = length(PN);
                
                x_mean = mean(N);                        
                x_t = repmat(x_mean,length(P),1);
                e = cdf*((1- cd*abs(x_t - P)./(abs(x_t)+abs(P))).*((x_t-P).^2));
                
                tmpp = dot(sum(isnan(e)),ones(1,d-1));
                if tmpp == 0
                    non_NaN = 1;
                end
                ze = zeros(length(N),d-1);
                e = [e; ze];
                xe = x_t-P;
                xe = [xe;ze];
            end
                
            ColOfOnes = ones(d-1,1);
            
            cvx_begin quiet                 
                variable w(d-1) 
                variable b;
                variable xi(n_build);
                variable t(n_build);
                variable u(n_build,d-1);
                variable v(n_build,d-1);

                minimize 1/2 *(norm(w)) + C*sum(xi);
                subject to
                    xi >= 0;
                    xi - 1 + PNL.*(PN*w+b)- t >= 0;
                    t - (u.*e)*ColOfOnes >= 0;
                    (v - u).*xe - 0.5*repmat((1+PNL),1,d-1).*repmat(w',n_build,1) == 0;
                    u >= 0;
                    v >= 0;       
            cvx_end
            
            svmc = fitcsvm(PN, PNL);
            
            %Ntr = reshape(Nt(randperm(length(Nt)*2)), length(Nt), 2);
            Ntr = zeros(length(Pt), d-1);
            for k=1:length(Pt)
                ra = randi([1 length(Nt)]);
                Ntr(k,:) = Nt(ra,:);
            end
            
            Ptattk = Pt+fa*(Ntr-Pt);
            X_test_attk = [Ptattk;Nt];
            y_test_attk = [ones(length(Ptattk),1); -1*ones(length(Nt),1)];
            
            ypred = Nt*w+b;
            correct = correct + sum(ypred<=0);
            ypred = Ptattk*w+b;
            correct = correct + sum(ypred>0);
            
            svmv = predict(svmc,X_test_attk);
            svmcorrect = svmcorrect + sum(y_test_attk==svmv);
        end
        output_adsvm(i,j) = (correct/round/n_test)*100;
        output_svm(i,j) = (svmcorrect/round/length(y_test_attk))*100;
    end
    fprintf('cd = %f done\n', cd);
end

T_output = array2table(output_adsvm);
T_output.Properties.VariableNames(1:5) = {'fa=0','fa=0.3','fa=0.5','fa=0.7','fa=1.0'};
writetable(T_output,'result/spam/restrained/output_accuracy.csv')
T_output = array2table(output_svm);
T_output.Properties.VariableNames(1:5) = {'fa=0.0','fa=0.3','fa=0.5','fa=0.7','fa=1.0'};
writetable(T_output,'result/spam/standard/output_accuracy_res.csv')