function [X,indices,y,yinst,ybag]=struct_to_concat_migraph(dataset)
        X=[];
        y=[];
        indices=[];
        yinst=[];

        ybag=zeros(length(dataset),1);
        for bb = 1:length(dataset)
            Nb=size(dataset{bb,1},1);
            indices=[indices; bb*ones(Nb,1)];
            y=[y; dataset{bb,2}*ones(Nb,1)];
            yinst=[yinst; dataset{bb,3}];
            X=[X; dataset{bb,1}];
            ybag(bb)=dataset{bb,2};
        end
end