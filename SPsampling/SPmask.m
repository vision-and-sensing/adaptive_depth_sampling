% parameters
samp_perc = 0.00125;

N = round(760*1280*samp_perc);   % num of samples
%thr = 100;           % threshold [m]
%sets = {'train','validation','test'};
sets = {'test'};

for set_ind = 1:length(sets) 
    set = sets{set_ind};
    % input
    RGB_path = fullfile("/home/tcenov/Depth/Datasets/Synthia/",set,"RGB");
    GT_path = fullfile("/home/tcenov/Depth/Datasets/Synthia/",set,"GT");

    % output
    sparse_path = fullfile("/home/tcenov/Depth/Datasets/Synthia/",set,"LiDAR",strcat("sp",num2str(samp_perc)));

    if ~exist(sparse_path, 'dir')
       mkdir(sparse_path)
    end
    
    imds = imageDatastore(RGB_path);
    tot_imgs=length(imds.Files);
    pix=zeros(tot_imgs,1);
    for k=1:tot_imgs
        [img,info] = readimage(imds,k);

        img_from_path=info.Filename;
        img_name=img_from_path(end-10:end);
        save_path=fullfile(sparse_path,img_name);

        if ~isfile(save_path)
            [~, SampMask] = mask_spSampling(img,N);

            gt_img = imread(fullfile(GT_path, img_name));
            sparse = gt_img .* uint16(SampMask);
            %sparse(gt_img > thr * 256) = 0;

            pix(k) = nnz(SampMask);
            imwrite(sparse,save_path)
            if mod(k-1,10) == 0
                disp([num2str(k/tot_imgs*100), '%'])
            end
        end
    end
    mean_pix = mean(pix);
    disp(['Mean Saples: ', num2str(mean_pix)])
end