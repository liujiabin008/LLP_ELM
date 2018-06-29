function [TrainingAccuracy, TestingAccuracy] = multi_class_llp(TrainingData,TrainingLabel, TrainingProp, TestingData,TestingLabel,Number_class,split)

T = TrainingLabel';
P = TrainingData';

TV.T = TestingLabel';
TV.P = TestingData';

lamda = 0.1;
u = 10^(-4);
p = 1.1;
u_max = 10^6;


NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofDimension=size(P,1);

C = Number_class; % total classes 
D =  NumberofDimension;  % total dimension
BAG_NUM  = size(TrainingProp,2);


P = ones(C,D);

for i = 1:BAG_NUM 
    Z(:,:,i) = TrainingData(:,find(split.train_bag_idx == i));
end


z = zeros(D,BAG_NUM);
for i = 1:BAG_NUM
    z(:,i) = sum(TrainingData(:,find(split.train_bag_idx == i)));
end






end


Y = ones(C,BAG_SIZE);


% ----- define matrix -----

A = lamda*ones(C,C);
B = zeros(D,D);
for i=1;BAG_NUM
    B = B + z(:,i)*z(:,i)' + u*Z(:,:,i)*Z(:,:,i)';
end

for i = 1;BAG_NUM
    
end

