function [SPim, SampMask] = mask_spSampling(rgbim,N)

h = size(rgbim,1);
w = size(rgbim,2);
Npix = h*w;

% [LabelSP,Nsp] = superpixels(rgbim,N,'Compactness',20);
[LabelSP,Nsp] = superpixels(rgbim,N,'Compactness',20,'Method','slic');
SPinds = cell(1,Nsp);
for s = 1:Nsp
    SPinds{s} = find(LabelSP==s);
end
SPbounadries = boundarymask(LabelSP);
SPim = imoverlay(rgbim,SPbounadries,'cyan');

SampMask = zeros(size(rgbim,1),size(rgbim,2));
for c = 1:Nsp
    inds = SPinds{c};
    % sample pixel which is closest to SP center of mass
    [subI,subJ] = ind2sub([h w],inds);
    avg_sub = round(mean([subI,subJ]));
    avg_sub_mat = repmat(avg_sub,length(inds),1);
    [~,avg_subI] = min(sum((subI-avg_sub_mat(:,1)).^2,2));
    [~,avg_subJ] = min(sum((subJ-avg_sub_mat(:,2)).^2,2));
    avg_subI = subI(avg_subI);
    avg_subJ = subJ(avg_subJ);
    SampMask(avg_subI, avg_subJ) = 1;
end
