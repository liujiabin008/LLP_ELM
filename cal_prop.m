function [ bag_prop ] = cal_prop(label, bag_idx, bagnum,class_size)
bag_prop = zeros(bagnum, class_size);


for i=1:bagnum,
    for j=1:size(label,1),
        for k=1:class_size
            if label(j)==k && bag_idx(j)==i,
                bag_prop(i,k) = bag_prop(i,k) + 1;
            end      
        end
    end
    bag_prop(i,:) = bag_prop(i,:)/(length(find(bag_idx==i)));
end
end

