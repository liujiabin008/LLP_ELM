function [bag_idx, bagnum] = split_dataset(label, bagsize)
% stream=RandStream('mt19937ar','Seed',2);%% generate stream for reproducibility of model
% set(stream,'Substream',1);
% RandStream.setGlobalStream(stream);
% data = ScaledMatrixByColumn(data,-1,1);
bagnum = ceil(length(label) / bagsize);
bagno = 1;
flag = 0;
bag_idx = zeros(length(label), 1);
for i=1:length(label),
    if flag == bagsize
        bagno = bagno + 1;
        bag_idx(i) = bagno;
        flag = 1;
    else
        bag_idx(i) = bagno;
        flag = flag + 1;
    end
end

rowrank = randperm(size(bag_idx, 1));
bag_idx = bag_idx(rowrank, :);

end
