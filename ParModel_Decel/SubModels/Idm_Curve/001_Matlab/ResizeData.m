function [Resized_Data] =ResizeData(Augmented_data, Resize)
j=1;
for i = 1: length(Augmented_data)
    if mod(i,Resize)==1
        Resized_Data(j) = Augmented_data(i);
        j = j+1;
    end
end
end