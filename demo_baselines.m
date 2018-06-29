clear all;
datasets1 = {'sonar', 'heart', 'vote(#435#16)', 'credit-a(#690#15)', 'diabetes', 'pima-indian(#768#8)','splice_scale(#1000#60)','breast-cancer','ala'};
datasets2 = {'Musk2(#6598#166)', 'magic', 'cod-rna(#59535#8)'};
datasets3 = {'shuttle_scale(#1000#9#7)','connect-4(#1000#126#3)','dna(#2000#180#3)','satimage(#4435#36#6)','protein(#1000#357#3)'};
datasets4 = {'data_101'};
fileID = fopen('result1.txt','w');

for i=1:1
fprintf(fileID,'%s\n', datasets4{i});
load(['./data_pic/', datasets4{i}, '.mat']);
data = ScaledMatrixByColumn(double(data),-1,1);
bagsizeset = [2,4,8,16,32,64];
%bagsizeset = [256,512,1024];
split_original = split;
class_size = max(split.train_label);

for b=1:6
bagsize = bagsizeset(b);
fprintf(fileID,'%d\n', bagsize);
indices=crossvalind('Kfold',size(data,1),5);

elm_accuracy=zeros(5,1);
elm_time=zeros(5,1);


for k=1:5
split = split_original;
split.test_data_idx = (indices == k); 
split.train_data_idx = ~split.test_data_idx;

teX = data(split.test_data_idx,:);
split.test_label = split.train_label(split.test_data_idx,:);
trX = data(split.train_data_idx,:);
split.train_label = split.train_label(split.train_data_idx,:);

[split.train_bag_idx, bagnumtr] = split_dataset(split.train_label, bagsize);
[split.test_bag_idx, bagnumte] = split_dataset(split.test_label, bagsize);

split.train_bag_prop = cal_prop(split.train_label, split.train_bag_idx, bagnumtr,class_size);
split.test_bag_prop = cal_prop(split.test_label, split.test_bag_idx, bagnumte,class_size);



%% ELM_LLP 
tic
BagNumber =  size(split.train_bag_prop,1); 
train_prob_test = split.train_bag_prop;

train_prob = zeros(BagNumber,class_size);
for i = 1:BagNumber
    train_prob(i,:) = train_prob_test(i,:)*length(find(split.train_bag_idx == i));
end
[TrainingAccuracy, TestingAccuracy] = elm_llp(data(split.train_data_idx,:), split.train_label, train_prob, data(split.test_data_idx,:), split.test_label, 3000, 'sig', split);
toc

elm_time(k) = toc




%%

elm_accuracy(k) = TrainingAccuracy;

end

elm_avg = mean(elm_accuracy)*100;
elm_std = std(elm_accuracy);
fprintf(fileID,'%.2f\\%%\\pm%.2f\n', elm_avg, elm_std);
fprintf(fileID,[num2str(mean(elm_time)),'s\n']);
end
end
fclose(fileID);