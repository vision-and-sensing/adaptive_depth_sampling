from __future__ import print_function
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import imageio
import os
from tqdm import tqdm
from func.utils import *
# Use to compare reconstruction results of different sampling patterns (grid, rand, sp, imp) under error 'err_type'.

# parameters
dataset_path = ""                   # <----- INSERT DEFAULT VALUE (dataset root dir)
samp_perc = 0.01                    # sampling percentage
sigma = 3.0                         # sigma
grd = 0.05                          # grid portion
rand_iter = 10                      # random masks
rand_batch_size = 10                # random masks per batch (high res 4, low res )
err_type = 'RMSE'                   # 'MSE' / 'RMSE' / 'PSNR' / 'REL' / 'MAD' / 'TUKEY' / 'same' for same as ImpMap

# save result images
save_all_imgs = False
save_some_imgs = False
upper_thr = 0.45                            # if save some images - save within this improvement over random
lower_thr = -0.20
save_image_list = ('5000286.png', '123')    # if save some images - save these images regardless of improvement

gt_fold = dataset_path + "/test/GT/"        # GT path
rgb_fold = dataset_path + "/test/RGB/"      # RGB path

rand_sparse_fold = dataset_path + "/test/LiDAR/rand" + str(samp_perc) + "/"                 # construct sparse folders paths
grid_sparse_fold = dataset_path + "/test/LiDAR/grid" + str(samp_perc) + "/"
sp_sparse_fold = dataset_path + "/test/LiDAR/sp" + str(samp_perc) + "/"
imp_sparse_fold = dataset_path + "/test/LiDAR/unnorm_0.25pow_rgb2imp_err" + err_type + "_perc" + str(samp_perc) + "_iter100_m50000.0_grd" + str(grd) + "_sig" + str(sigma) + "/"

rand_model_path, grid_model_path, sp_model_path, imp_model_path = net_paths(samp_perc, err_type, sigma, grd)    # read models paths (as saved in models bank in utils)

rand_model = fusionnet_def(rand_model_path)             # read models
grid_model = fusionnet_def(grid_model_path)
sp_model = fusionnet_def(sp_model_path)
imp_model = fusionnet_def(imp_model_path)


def main():
    font = {'family': 'normal',                         # fonts for plot
            'weight': 'bold',
            'size': 14}
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rc('font', **font)

    image = [img for img in os.listdir(rgb_fold)]       # read rgb images list
    left_test = [rgb_fold + img for img in image]
    left_test.sort()

    imp_to_rand = []                                    # initialize parameters
    imp_to_grid = []
    imp_to_sp = []
    max_err = 0
    rand_errors = []
    grid_errors = []
    sp_errors = []
    imp_errors = []
    bad_imgs_list = []
    d_mean = []
    d_var = []

    err_measure = def_error(err_type)                   # define error metric

    fig, (ax1, ax2) = plt.subplots(2, figsize=(17, 15))         # define figure
    fig.suptitle(u'Sampling Budget:' + str(samp_perc*100.0) + '%, \u03C3:' + str(sigma) + ', grid' + str(grd))

    for inx in tqdm(range(len(left_test))):                 # for each RGB image
        im_name = left_test[inx].split('/')[-1]                 # read file name

        if im_name not in bad_imgs:                             # if not a bad image (badly rendered)
            imgL = skimage.io.imread(rgb_fold + im_name)            # read RGB image
            imgL = np.transpose(imgL, (2, 0, 1))
            imgL = np.expand_dims(imgL, axis=0)

            gtruth = skimage.io.imread(gt_fold + im_name, as_gray=True) # read GT image
            gtruth = gtruth / 256.0

            grid_sparse = skimage.io.imread(grid_sparse_fold + im_name, as_gray=True)   # read grid sparse
            grid_sparse = grid_sparse / 256.0

            sp_sparse = skimage.io.imread(sp_sparse_fold + im_name, as_gray=True)       # read superpixels sparse
            sp_sparse = sp_sparse / 256.0

            imp_sparse = skimage.io.imread(imp_sparse_fold + im_name, as_gray=True)     # read importance sparse
            imp_sparse = imp_sparse / 256.0

    ######################################## random sample
            rand_err = []
            inp_counter = 0
            for r in range(rand_iter):                                      # for rand_iter iterations
                inp_counter += 1
                rand_mask = build_rand_mask(imgL.shape[-2:], samp_perc)         # build a random mask
                rand_sparse = rand_mask * gtruth                                # sample depth
                rand_sparse = np.expand_dims(np.expand_dims(rand_sparse, 0), 0)
                if inp_counter == 1:                                            # initialize batch
                    sparse_in = rand_sparse
                    imgL_in = imgL
                else:                                                           # concat to batch
                    sparse_in = np.concatenate((sparse_in, rand_sparse))
                    imgL_in = np.concatenate((imgL_in, imgL))
                if inp_counter == rand_batch_size or r == rand_iter - 1:        # correct batch size
                    rand_pred = fusionnet_inp(rand_model, imgL_in, sparse_in)   # input to model
                    num_of_preds = rand_pred.shape[0]
                    for p in range(num_of_preds):                               # for each prediction
                        rand_err.append(err_measure(gtruth, rand_pred[p, 0, :, :])) # calculate errors
                    inp_counter = 0

    ######################################## grid sample
            grid_mask = np.uint8(grid_sparse > 0)
            grid_sparse = np.expand_dims(np.expand_dims(grid_sparse, 0), 0)
            grid_pred = fusionnet_inp(grid_model, imgL, grid_sparse)
            grid_err = err_measure(gtruth, grid_pred[0, 0, :, :])

    ######################################## sp sample
            sp_mask = np.uint8(sp_sparse > 0)
            sp_sparse = np.expand_dims(np.expand_dims(sp_sparse, 0), 0)
            sp_pred = fusionnet_inp(sp_model, imgL, sp_sparse)
            sp_err = err_measure(gtruth, sp_pred[0, 0, :, :])

    ######################################## imp sample
            imp_mask = np.uint8(imp_sparse > 0)
            imp_sparse = np.expand_dims(np.expand_dims(imp_sparse, 0), 0)
            imp_pred = fusionnet_inp(imp_model, imgL, imp_sparse)
            imp_err = err_measure(gtruth, imp_pred[0, 0, :, :])

    ######################################## plot
            impr0 = (rand_err[0] - imp_err) / rand_err[0]           # improvement of importance over first random
            if save_all_imgs or (save_some_imgs and ((impr0 > upper_thr or impr0 < lower_thr) or im_name in save_image_list)):      # if in range
                print('Improvement: ' + str(round(impr0 * 100, 2)) + '[%]. Saving images...')
                save_res_imgs(gtruth, imgL, rand_pred, grid_pred, sp_pred, imp_pred, imp_sparse_fold, impr0,    # save images
                              rand_mask, grid_mask, sp_mask, imp_mask,
                              rand_sparse, grid_sparse, sp_sparse, imp_sparse,
                              err_type, rand_err, grid_err, sp_err, imp_err, im_name, inx)

            # impr = (np.mean(rand_err) - imp_err) / np.mean(rand_err)    # mean improvement

            rand_errors.append(np.mean(rand_err))                       # save errors
            grid_errors.append(grid_err)
            sp_errors.append(sp_err)
            imp_errors.append(imp_err)

            imp_to_rand.append((np.mean(rand_err)-imp_err)/np.mean(rand_err))   # improvement ofer random, grid and SP
            imp_to_grid.append((grid_err-imp_err)/grid_err)
            imp_to_sp.append((sp_err-imp_err)/sp_err)

            d_mean.append(np.mean(gtruth[gtruth <= depth_thr]))                 # depth mean and var
            d_var.append(np.var(gtruth[gtruth <= depth_thr]))

            ax1.scatter((inx - len(bad_imgs_list) + 1) * np.ones(rand_iter), rand_err, s=1, c=np.array([0, 0, 1.0]))    # add scatter plot of results for this image
            ax1.scatter((inx - len(bad_imgs_list) + 1), grid_err, s=1, c=np.array([0, 1.0, 0]))
            ax1.scatter((inx - len(bad_imgs_list) + 1), sp_err, s=1, c=np.array([0.5, 0, 1.0]))
            ax1.scatter((inx - len(bad_imgs_list) + 1), imp_err, s=1, c=np.array([1.0, 0, 0]))

            max_err_in_iter = np.max((grid_err, sp_err, imp_err, np.max(rand_err)))         # save max error (for ylim)
            if max_err_in_iter > max_err:
                max_err = max_err_in_iter

        else:                               # badly rendered image
            print('Bad image: ' + im_name)
            bad_imgs_list.append(im_name)

    st_rand = 'Mean Rand error: ' + str(round(np.mean(rand_errors), 4))         # construct string for print
    st_grid = 'Mean Grid error: ' + str(round(np.mean(grid_errors), 4))
    st_sp = 'Mean SP error: ' + str(round(np.mean(sp_errors), 4))
    st_imp = 'Mean Imp error: ' + str(round(np.mean(imp_errors), 4))
    print(st_rand)
    print(st_grid)
    print(st_sp)
    print(st_imp)

    if err_type != 'PSNR':                                                                  # improvement (in percents) over other sampling patterns
        stats_rand = str(round(np.mean(imp_to_rand) * 100, 2)) + '% lower than random'
        stats_grid = str(round(np.mean(imp_to_grid) * 100, 2)) + '% lower than grid'
        stats_sp = str(round(np.mean(imp_to_sp) * 100, 2)) + '% lower than superpixels'
    else:
        stats_rand = str(-1 * (round(np.mean(imp_to_rand) * 100, 2))) + '% higher than random'
        stats_grid = str(-1 * (round(np.mean(imp_to_grid) * 100, 2))) + '% higher than grid'
        stats_sp = str(-1 * (round(np.mean(imp_to_sp) * 100, 2))) + '% higher than superpixels'

    fontsize = 20                               # add information to scatter plot
    ax1.grid()
    ax1.legend(('Random', 'Grid', 'SuperPixels', 'Importance'), loc=2, markerscale=4)
    ax1.axis(xmin=0, xmax=len(imp_to_rand) + 1)
    ax1.set_xlabel('Test set image index', fontsize=fontsize, weight='bold')
    ax1.set_title((stats_rand + '\n' + stats_grid + '\n' + stats_sp))
    if err_type == 'RMSE':
        ax1.set_ylabel(err_type + ' [m]', fontsize=fontsize, weight='bold')
    else:
        ax1.set_ylabel(err_type, fontsize=fontsize, weight='bold')
    if err_type == 'REL':
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    imp_to_rand_100 = [val * 100.0 for val in imp_to_rand]                      # plot improvement (for each image)
    ax2.plot(imp_to_rand_100)
    ax2.axis(xmin=0, xmax=len(imp_to_rand) + 1)
    ax2.set_xlabel('Test set image index', fontsize=fontsize, weight='bold')
    ax2.set_ylabel('Improvement [%]', fontsize=fontsize, weight='bold')
    ax2.grid()

    plt.savefig(os.path.join(os.path.dirname(imp_model_path), 'res.png'), dpi=500, transparent=True)    # save result plot
    f = open(os.path.join(os.path.dirname(imp_model_path), 'stats.txt'), "w+")
    f.write(st_rand + '\n')
    f.write(st_grid + '\n')
    f.write(st_sp + '\n')
    f.write(st_imp + '\n')
    f.close()

    print('Bad images list: ')
    print(bad_imgs_list)
    print('Total bad imgs: ' + str(len(bad_imgs_list)))

    plt.show()

    # samp_loc = mark_samp_patterns(np.transpose(imgL[0], (1, 2, 0)), rand_mask, grid_mask, sp_mask, imp_mask)
    # plt.imshow(mark_samp_patterns(np.transpose(imgL[0],(1,2,0)), rand_mask,  grid_mask, sp_mask, imp_mask)) # blue, green, black, red


if __name__ == '__main__':
    main()





