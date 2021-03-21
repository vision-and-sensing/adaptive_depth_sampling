from tqdm import tqdm
import skimage.io
import imageio
import argparse
from func.utils import *
# Gaussian Sampling - calculate sampling pattern based on importance maps and save the sampled sparse depth

# parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='', help='dataset root path')     # <----- INSERT DEFAULT VALUE (dataset root dir)
parser.add_argument('--impmaps_in', type=str, default='', help='impmaps in input')     # <----- INSERT DEFAULT VALUE (impmaps name in dataset)
parser.add_argument('--sigmas', nargs='*', default=[1.0, 2.0, 3.0], help='a list of sigmas to use (in iterations)')
parser.add_argument('--samp', type=float, default=0.01, help='sampling percentage (0.01 for 1%)')
parser.add_argument('--unnorm', type=bool, default=False, help='unnormalize impmaps (rise to 4 power) before calculation (if im2im network was trained on impmaps**0.25)')
parser.add_argument('--frame_pix', type=int, default=8, help='width of ignored outer frame of impmap (artifacts)')
parser.add_argument('--grd', type=float, default=0.05, help='portion of --samp as grid (0.05 default)')
args = parser.parse_args()

# other parameters
sets = ('train', 'validation', 'test')      # sets within the dataset to calculate
grd = args.grd                              # other initialization
samp_perc = args.samp
saved_samp_perc = samp_perc

for param in args.sigmas:                   # for each value of sigma

    if grd > 0:                             # print grid portion (5% in default)
        print("grid portion: " + str(grd))

    print('sigma: ' + str(param))           # print current sigma
    for set in sets:                        # for each sets in dataset
        print('set: ' + set)                # print current set

        gt_fold = os.path.join(args.dataset_path, set, "GT/")                                   # Input: current GT folder
        impmap_fold = os.path.join(args.dataset_path, set, "ImpMaps/" + args.impmaps_in + "/")  # Input: current ImpMaps folder

        fold_name = args.impmaps_in                                                             # output folder name construction
        if fold_name.split('perc')[1].split('_')[0] != samp_perc:                               # if current sampling percentage is different from the one of ImpMaps calculation
            fold_name = fold_name + '_PERC' + str(samp_perc)                                        # add new sampling percentage
        if grd > 0:                                                                             # if there is a grid portion
            fold_name = fold_name + '_grd' + str(grd)                                               # add grid portion
        if args.unnorm:                                                                         # if unnorm
            fold_name = 'unnorm_' + fold_name                                                       # add flag
        fold_name = fold_name + "_sig" + str(param)                                             # add current sigma

        sparse_fold = os.path.join(args.dataset_path, set, "LiDAR", fold_name)                  # final output ImpMap folder
        if not os.path.exists(sparse_fold):
            os.mkdir(sparse_fold)

        ImpMaps = [img for img in os.listdir(impmap_fold)]                                      # construct a list of ImpMaps in input
        ImpMap_test = [impmap_fold + img for img in ImpMaps]
        ImpMap_test.sort()

        impmap_init = skimage.io.imread(ImpMap_test[0])                                         # construct frame where ImpMap will be zero (due to artifacts in generation)
        frame = np.ones((impmap_init.shape[0], impmap_init.shape[1]))
        frame[args.frame_pix:-args.frame_pix, args.frame_pix:-args.frame_pix] = 0
        frame = frame > 0

        N = round(saved_samp_perc * impmap_init.shape[0] * impmap_init.shape[1])                # number of samples
        res = impmap_init.shape                                                                 # resolution
        tot_pix = res[0] * res[1]                                                               # total number of pixels

        for inx in tqdm(range(len(ImpMap_test))):                                               # for each ImpMap
            im_name = ImpMap_test[inx].split('/')[-1].split('_')[0]                                 # read file name
            sparse_savepath = os.path.join(sparse_fold, im_name)                                    # construct output file path

            if not os.path.isfile(sparse_savepath):                                                 # if does not exists
                gtruth = skimage.io.imread(gt_fold + im_name, as_gray=True)                             # read GT image (has to be divided by 256 to convert to meters)
                impmap = skimage.io.imread(impmap_fold + im_name).astype('float64') / 256.0             # read ImpMap

                samp_perc = saved_samp_perc                                                             # initialize sampling percentage

                impmap[frame] = 0.0                                                                     # zero ImpMap in frame (edges) to ignore generation artifacts

                if args.unnorm:                                                                         # unnormalize if needed (rise to 4 power)
                    impmap = (((impmap / 64.0) ** 2.0) ** 2.0)

                if np.count_nonzero(impmap) < N * (1 - grd):                                            # if there is not enough non-zero values in ImpMap for Gaussian Sampling
                    imp_mask = (impmap != 0) * 1.0                                                           # select all non-zero

                    temp_grd = grd                                                                          # initialize grid sampling portion
                    while 1:
                        grd_mask = build_grid_mask(res, samp_perc * temp_grd)                                   # build grid mask in resolution 'res' and portion 'total budget' * 'grid portion'
                        if np.count_nonzero(imp_mask + grd_mask) < N:                                           # if important samples (all impmap in this case) + grid does not cover all budget
                            temp_grd *= 1.01                                                                        # increase grid sampling portion
                        else:                                                                                   # else (enough samples)
                            imp_mask += grd_mask                                                                    # and two patterns and break
                            break

                else:                                                                                   # else (enough non-zero in ImpMap)
                    imp_mask = np.zeros_like(impmap)                                                        # initialize mask
                    if grd > 0:                                                                             # if there is a grid portion
                        grd_mask = build_grid_mask(res, samp_perc * grd)                                        # construct grid mask
                        imp_mask += grd_mask                                                                    # add it to total mask
                        impmap[grd_mask != 0] = 0.0                                                             # zero ImpMap in grid pattern points (since they were already selected)
                        N_grd = np.count_nonzero(grd_mask)                                                      # count samples in grid
                        grd_samp_perc = N_grd / tot_pix                                                         # count portion of grid
                        samp_perc -= grd_samp_perc                                                              # decrease it from total sampling portion

                    imp_mask += pick_max(impmap, samp_perc, param)                                          # Gaussian Sampling
                    impmap[imp_mask != 0.0] = 0.0                                                           # zero ImpMap in selected points

                    try_c = 0                                                                               # initialize counter
                    while np.count_nonzero(imp_mask) < N and (np.any(impmap) and try_c < 5):                # if there is still not enough samples (less then the budget)
                        print('Need ' + str(N - np.count_nonzero(imp_mask)) + ' more samples... (try ' + str(try_c + 1) + ')')  # print a notification
                        impmap[imp_mask != 0.0] = 0.0                                                       # zero ImpMap in selected points
                        N_samp = np.count_nonzero(imp_mask)                                                 # number of samples in mask
                        cur_samp_perc = N_samp / tot_pix                                                    # portion of samples from entire image
                        imp_mask += pick_max(impmap, saved_samp_perc - cur_samp_perc, param)                # Gaussian Sampling
                        try_c += 1                                                                          # increase counter

                imp_sparse = np.where(imp_mask != 0, gtruth, 0.0)                                           # sample depth (sparse depth map)

                if np.count_nonzero(imp_sparse) != N:                                                       # if there is still not enough sampels
                    print('Num of samples in image ' + im_name + ": " + str(np.count_nonzero(imp_sparse)) + ' instead of ' + str(N))    # print a notification

                imageio.imwrite(sparse_savepath, imp_sparse.astype('uint16'))                               # save the result sparse depth map




