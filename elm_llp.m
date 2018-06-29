function [TrainingAccuracy, TestingAccuracy] = elm_llp(TrainingData,TrainingLabel, TrainingProp, TestingData,TestingLabel, NumberofHiddenNeurons, ActivationFunction, split)

% Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%                           'tribas' for Triangular basis function
%                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)


%%%%%%%%%%% Load  dataset

T = TrainingLabel';
P = TrainingData';

TV.T = TestingLabel';
TV.P = TestingData';


NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);


%%%%%%%%%%%% Preprocessing the data of classification
sorted_target=sort(cat(2,T,TV.T),2);
label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
label(1,1)=sorted_target(1,1);
j=1;
for i = 2:(NumberofTrainingData+NumberofTestingData)
    if sorted_target(1,i) ~= label(1,j)
        j=j+1;
        label(1,j) = sorted_target(1,i);
    end
end
number_class=j;
NumberofOutputNeurons=number_class;
       
%%%%%%%%%% Processing the targets of training
temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
for i = 1:NumberofTrainingData
    for j = 1:number_class
        if label(1,j) == T(1,i)
             break; 
        end
    end
    temp_T(j,i)=1;
end

%T=temp_T*2-1;
T = temp_T;

%%%%%%%%%% Processing the targets of testing
temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
for i = 1:NumberofTestingData
    for j = 1:number_class
        if label(1,j) == TV.T(1,i)
             break; 
        end
    end
    temp_TV_T(j,i)=1;
end
%TV.T=temp_TV_T*2-1;
TV.T=temp_TV_T;

%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

NumberofBag = length(split.train_bag_prop);

H_Bag = zeros(NumberofHiddenNeurons,NumberofBag);

for i = 1:NumberofBag
    H_Bag(:,i) =  sum(H(:,find(split.train_bag_idx == i)),2);
end



%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
%OutputWeight=pinv(H_Bag') * TrainingProp*NumberofBag;                       

OutputWeight= H_Bag*inv(eye(size(H_Bag',1))/0.1 + H_Bag'*H_Bag)*TrainingProp;


end_time_train=cputime;
TrainingTime=end_time_train-start_time_train        %   Calculate CPU time (seconds) spent for training ELM

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data

clear H_Bag;
clear H;

%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);        
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data



%%%%%%%%%% Calculate training & testing classification accuracy
MissClassificationRate_Training=0;
MissClassificationRate_Testing=0;

for i = 1 : size(T, 2)
    [x, label_index_expected]=max(T(:,i));
    [x, label_index_actual]=max(Y(:,i));
    if label_index_actual~=label_index_expected
        MissClassificationRate_Training=MissClassificationRate_Training+1;
    end
end
TrainingAccuracy=1-MissClassificationRate_Training/size(T,2)


for i = 1 : size(TV.T, 2)
    [x, label_index_expected]=max(TV.T(:,i));
    [x, label_index_actual]=max(TY(:,i));
     if label_index_actual~=label_index_expected
         MissClassificationRate_Testing=MissClassificationRate_Testing+1;
     end
end
TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)  