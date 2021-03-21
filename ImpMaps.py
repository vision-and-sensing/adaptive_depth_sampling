from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import random
import skimage.io
import imageio
import time
import os
import argparse
from tqdm import tqdm
from func.utils import *
# Use to create Importance Maps under error 'err_type'

# parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='', help='dataset root path')               # <----- INSERT DEFAULT VALUE (dataset root dir)
parser.add_argument('--model_path', type=str, default='', help='FusionNet trained model path')      # <----- INSERT DEFAULT VALUE (trained model path)
parser.add_argument('--samp_perc', type=float, default=0.01, help='sampling percentage (0.01 for 1%)')
parser.add_argument('--mask_batch_size', type=int, default=5, help='mask batch size')
parser.add_argument('--rand_iters', type=int, default=100, help='how many inputs (and predictions) to use in the calculation of ImpMaps')
parser.add_argument('--err_type', type=str, default='RMSE', help='error type: RMSE / REL (all options: MSE / RMSE / PSNR / REL / MAD / TUKEY)')
parser.add_argument('--mult_factor', type=float, default=None, help='# scale impmap before saving (None for automatic (RMSE and REL))')
args = parser.parse_args()

# other parameters
sets = ('train', 'validation', 'test')      # sets within the dataset to calculate


def calc_impmap(model, imgL, gt, iters=args.rand_iters):

    imgL = np.transpose(imgL, (2, 0, 1))                    # reshape RGB (for concat)
    imgL = np.expand_dims(imgL, axis=0)

    ImpMap = np.zeros((imgL.shape[2], imgL.shape[3]))       # initialize ImpMap

    valid_map = (gt <= depth_thr) * (gt > 0)                # find valid region (0 < GT <= depth_thr)

    inp_counter = 0                                         # initialize input counter (single) and batch counter
    batch = 0

    for it in range(iters):                                 # for number of iterations (different inputs in ImpMap calculation)
        inp_counter += 1                                    # increase counter of inputs
        mask, _ = rand_mask_max(valid_map * 1.0, args.samp_perc)    # calculate a random sampling mask in valid region
        sparse = np.zeros_like(gt)                          # initialize sparse depth mask
        sparse[mask > 0] = gt[mask > 0]                     # sample depth in sampling mask points
        sparse = np.expand_dims(np.expand_dims(sparse, 0), 0)   # expand (for concat)

        if inp_counter == 1:                                # for first input initialize batch of RGB and sparse depth
            sparse_in = sparse
            imgL_in = imgL
        else:                                               # for other inputs concat RGB and sparse depth to batch
            sparse_in = np.concatenate((sparse_in, sparse))
            imgL_in = np.concatenate((imgL_in, imgL))

        if inp_counter == args.mask_batch_size or it == iters - 1:  # if coorect stack size (mask_batch_size or last)
            num_of_preds = sparse_in.shape[0]               # number of items in stack
            batch += 1                                      # increase batch counter

            pred = fusionnet_inp(model, imgL_in, sparse_in)     # use model to predict dense depth for batch

            for p in range(num_of_preds):                   # for each output calculate per-pixel error and add to ImpMap
                if args.err_type == 'RMSE':
                    ImpMap[valid_map] += (gt[valid_map] - pred[p, 0][valid_map]) ** 2
                if args.err_type == 'REL':
                    ImpMap[valid_map] += np.abs(gt[valid_map] - pred[p, 0][valid_map])/(gt[valid_map])
            inp_counter = 0                                 # zero input counter for next batch

    ImpMap /= iters                                         # divide the ImpMap by number of outputs (so that ImpMap is mean per-pixel error)
    return ImpMap


def main():
    if mult_factor is None:                     # define scaling parameter for ImpMaps (to extend the dynamic range to 2^16)
        if args.err_type == 'RMSE':
            mult_factor = 100.0
        if args.err_type == 'REL':
            mult_factor = 50000.0

    model = fusionnet_def(args.model_path)      # define FusionNet model

    impmap_type = "err" + args.err_type + "_perc" + str(args.samp_perc) + "_iter" + str(args.rand_iters) + '_m' + str(mult_factor)  # define ImpMaps folder name
    for set1 in sets:
        print('set: ' + set1)

        gt_fold = os.path.join(args.dataset_path, set1, 'GT') + "/"             # input GT folder of the set
        rgb_fold = os.path.join(args.dataset_path, set1, 'RGB') + "/"           # input RGB folder of the set

        impmap_fold = os.path.join(args.dataset_path, set1, 'ImpMaps') + "/"    # output root ImpMaps folder
        if not os.path.exists(impmap_fold):
            os.mkdir(impmap_fold)

        gt = [img for img in os.listdir(gt_fold)]                               # make a list of paths to images in GT folder
        image = [img for img in os.listdir(rgb_fold)]                           # make a list of paths to images in RGB folder

        gt_test = [gt_fold + img for img in gt]
        left_test = [rgb_fold + img for img in image]

        gt_test.sort()
        left_test.sort()

        ImpMap_path = os.path.join(impmap_fold, impmap_type)                    # output ImpMaps folder
        if not os.path.exists(ImpMap_path):
            os.mkdir(ImpMap_path)

        for inx in tqdm(range(len(left_test))):                                 # for each RGB + GT couple
            output1 = '' + left_test[inx].split('/')[-1]                        # read image name
            ImpMap_savepath = os.path.join(ImpMap_path, output1)                # construct a saving path for the specific ImpMap

            if not os.path.isfile(ImpMap_savepath):                             # if this file does not already exists
                assert left_test[inx].split('/')[-1] == gt_test[inx].split('/')[-1]     # make sure to read similarly named RGB and GT

                imgL = skimage.io.imread(left_test[inx])                        # read RGB image
                gtruth = skimage.io.imread(gt_test[inx], as_gray=True)          # read GT image
                gtruth = gtruth / 256.0                                         # normalize GT image

                ImpMap = calc_impmap(model, imgL, gtruth)                       # for this RGB and GT couple calculate an ImpMap

                ImpMap *= mult_factor                                           # scale ImpMap before saving
                ImpMap = np.clip(ImpMap, 0, 2**16 - 1)                          # clip ImpMap to the saved range

                imageio.imwrite(ImpMap_savepath, ImpMap.astype('uint16'))       # save ImpMap


if __name__ == '__main__':
    main()
