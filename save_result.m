
T_output = array2table(output_svm);
T_output.Properties.VariableNames(1:5) = {'fa=0.0','fa=0.3','fa=0.5','fa=0.7','fa=1.0'};
writetable(T_output,'result/spam/standard/output_accuracy.csv')
%{
T_wtrain = array2table(output_wtrain);
T_wtrain.Properties.VariableNames(1:2) = {'w1', 'w2'};
writetable(T_wtrain,'result/2D/free_range/w_train.csv')

T_btrain = array2table(output_btrain);
T_btrain.Properties.VariableNames(1) = {'b'};
writetable(T_btrain,'result/2D/free_range/b_train.csv')
%}
%{
train_T = readtable('data/train_data.csv','ReadVariableNames',false);
test_T = readtable('data/test_data.csv','ReadVariableNames',false);

X_train = train_T{:,1:2};
X_test = test_T{:,1:2};

scatter(X_train(:,1),X_train(:,2));
saveas(gcf,'img/2D/train.png');
scatter(X_test(:,1),X_test(:,2));
saveas(gcf,'img/2D/test.png');
%}