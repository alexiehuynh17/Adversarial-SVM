train_T = readtable('data/spam_train.csv','ReadVariableNames',false);
test_T = readtable('data/spam_test.csv','ReadVariableNames',false);

d = length(train_T.Properties.VariableNames);
X_train = train_T{:,1:d-1};
y_train = train_T{:,d};
n_train = length(y_train);
    
X_test = test_T{:,1:d-1};
y_test = test_T{:,d};
n_test = length(y_test);
Nt = X_test((y_test == -1),:);
Pt = X_test((y_test == 1),:);

f_attack = [0, 0.3, 0.5, 0.7, 1.0];
fa_len = length(f_attack);

output_svm = zeros(1, fa_len);
round = 10;

for j=1:fa_len
    fa = f_attack(j);
    fprintf('fa = %f\n', fa);
    svmcorrect = 0;
    for r=1:round
        rows = randperm(n_train, 0.1*n_train); 
        X_build = X_train(rows, :);
        y_build = y_train(rows, 1);

        N = X_build((y_build == -1),:);
        P = X_build((y_build == 1),:);
        PN = [P; N];
        PNL = [ones(length(P),1); -1*ones(length(N),1)];
        n_build = length(PN);
        
        svmc = fitcsvm(PN, PNL);
        
        Ntr = zeros(length(Pt), d-1);
        for k=1:length(Pt)
            ra = randi([1 length(Nt)]);
            Ntr(k,:) = Nt(ra,:);
        end
        Ptattk = Pt+fa*(Ntr-Pt);
        X_test_attk = [Ptattk;Nt];
        y_test_attk = [ones(length(Ptattk),1); -1*ones(length(Nt),1)];

        svmv = predict(svmc,X_test_attk);

        svmcorrect = svmcorrect + sum(y_test_attk==svmv);
    end
    output_svm(1,j) = (svmcorrect/round/length(y_test_attk))*100;
    fprintf('fa = %f done\n', fa);  
end

mw = (svmc.Alpha' * full(svmc.SupportVectors));
mbias = svmc.Bias;

T_output = array2table(output_svm);
T_output.Properties.VariableNames(1:5) = {'fa=0.0','fa=0.3','fa=0.5','fa=0.7','fa=1.0'};
writetable(T_output,'result/spam/standard/output_accuracy.csv')

%{
T_wtrain = array2table(mw);
T_wtrain.Properties.VariableNames(1:2) = {'w1', 'w2'};
writetable(T_wtrain,'result/2D/standard/w_train.csv')

T_btrain = array2table(mbias);
T_btrain.Properties.VariableNames(1) = {'b'};
writetable(T_btrain,'result/2D/standard/b_train.csv')
%}