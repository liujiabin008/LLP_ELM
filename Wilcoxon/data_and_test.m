%% wilcoxon test for Adaboost-LLP and baselines
%% LINEAR
% THE 5-FOLD CROSS VALIDATION RESULTS IN THE CASE OF LINEAR KERNEL.
% Dataset            Method           2 4 8 16 32 64
% sonar              MeanMap
% heart              InvCal
% vote               alter-∝SVM
% australian         p-NPSVM1
% breast-w           p-NPSVM2
% ionosphere
% breast-cancer
% credit-a
% pima-indian
% splice
% ala

load data.mat 


data_1 = [data(1:3,1) data(4:6,1)  data(7:9,1)  data(10:12,1) data(13:15,1)  data(16:18,1)];
data_2 = [data(1:3,2) data(4:6,2)  data(7:9,2)  data(10:12,2) data(13:15,2)  data(16:18,2)];
data_3 = [data(1:3,3) data(4:6,3)  data(7:9,3)  data(10:12,3) data(13:15,3)  data(16:18,3)];
data_4 = [data(1:3,4) data(4:6,4)  data(7:9,4)  data(10:12,4) data(13:15,4)  data(16:18,4)];
data_5 = [data(1:3,5) data(4:6,5)  data(7:9,5)  data(10:12,5) data(13:15,5)  data(16:18,5)];
data_6 = [data(1:3,6) data(4:6,6)  data(7:9,6)  data(10:12,6) data(13:15,6)  data(16:18,6)];
data_7 = [data(1:3,7) data(4:6,7)  data(7:9,7)  data(10:12,7) data(13:15,7)  data(16:18,7)]; 
data_8 = [data(1:3,8) data(4:6,8)  data(7:9,8)  data(10:12,8) data(13:15,8)  data(16:18,8)];
data_9 = [data(1:3,9) data(4:6,9)  data(7:9,9)  data(10:12,9) data(13:15,9)  data(16:18,9)];
									
% data_1 =[
%     0.6386	0.6568	0.6495	0.5916	0.5979	0.56
%     0.7371	0.6359	0.6055	0.5475	0.5537	0.547
%     0.7667	0.6632	0.52	0.5621	0.4737	0.4926
%     ];




alpha = 0.05;
k=1;
InvCal = [data_1(k,:);data_2(k,:);data_3(k,:);data_4(k,:);data_5(k,:);data_6(k,:);data_7(k,:);data_8(k,:);data_9(k,:)];
k=2;
alter_pSVM = [data_1(k,:);data_2(k,:);data_3(k,:);data_4(k,:);data_5(k,:);data_6(k,:);data_7(k,:);data_8(k,:);data_9(k,:)];
%k=3;
%con_pSVM = [data_1(k,:);data_2(k,:);data_3(k,:);data_4(k,:);data_5(k,:);data_6(k,:);data_7(k,:);data_8(k,:);data_9(k,:)];
k=3;
LLP_ELM = [data_1(k,:);data_2(k,:);data_3(k,:);data_4(k,:);data_5(k,:);data_6(k,:);data_7(k,:);data_8(k,:);data_9(k,:)];
% k=5;
% p_NPSVM2 = [data_1(k,:);data_2(k,:);data_3(k,:);data_4(k,:);data_5(k,:);data_6(k,:);...
%                 data_7(k,:);data_8(k,:);data_9(k,:);data_10(k,:);data_11(k,:)];


% 全部结果一起比较
all_data = [InvCal(:) alter_pSVM(:) LLP_ELM(:)];
[f_all,R_pos,p] = Wilcoxon(all_data,alpha);
n = numel(LLP_ELM);
total_pair = n*(n+1)/2;
R_neg = total_pair - R_pos;
R = [R_pos(end,1:end-1)' R_neg(end,1:end-1)' p(end,1:end-1)'];
% 对每个数据集比较
% f_each = cell(size(bl_1,1),1);
% for i = 1 : size(bl_1,1)
%     f_each{i} = Wilcoxon([bl_1(i,:)' bl_2(i,:)' bl_3(i,:)' bl_4(i,:)'],alpha);
% end

% addpath huatu;
% labels={'MeanMap','InvCal','alter-pSVM','p-NPSVM1','p-NPSVM2'};
% alpha=0.05;
% cd = criticaldifference(-all_data,labels,alpha);
% print('cd_diagram.eps','-depsc');
% t=1;
disp(R)
