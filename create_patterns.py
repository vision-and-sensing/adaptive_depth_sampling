import skimage.io
import argparse
from tqdm import tqdm
from func.utils import *
# create sampling patterns (random / grid) at different sampling budgets

# parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='', help='dataset root path')     # <----- INSERT DEFAULT VALUE (dataset root dir)
parser.add_argument('--pattern', type=str, default='rand', help='pattern selection rand / grid')
parser.add_argument('--samp', type=float, default=0.01, help='sampling percentage (0.01 for 1%)')
args = parser.parse_args()

# other parameters
limit_depth = False                     # limit depth sampling (at depth_thr). These samples will be removed on input (so it is advised to leave it at False).
sets = ('train', 'validation', 'test')  # list of sets for which to make patterns (train, validation, test)

# start calculation
print("Calculating " + args.pattern + str(args.samp))
for set1 in sets:
    print(set1)
    gt_fold = os.path.join(args.dataset_path, set1, "GT/")                           # GT dir path to sample from

    lidar_root_path = os.path.join(args.dataset_path, set1, "LiDAR")                 # LiDAR path folder (that contains all patterns)
    if not os.path.exists(lidar_root_path):
        os.mkdir(lidar_root_path)

    sparse_fold = os.path.join(lidar_root_path, args.pattern + str(args.samp))  # LiDAR subfolder with the specific pattern
    if not os.path.exists(sparse_fold):
        os.mkdir(sparse_fold)

    gt = [img for img in os.listdir(gt_fold)]                                   # make a list of paths to images in GT folder
    gt_test = [gt_fold + img for img in gt]
    gt_test.sort()

    for inx in tqdm(range(len(gt_test))):                                       # for each image in list
        im_name = gt_test[inx].split('/')[-1]                                   # read its name
        sparse_savepath = os.path.join(sparse_fold, im_name)                    # construct a saving path (including name)

        if not os.path.isfile(sparse_savepath):                                 # if it does not already exists
            gtruth = skimage.io.imread(gt_fold + im_name, as_gray=True)         # read GT image to sample from
            if args.pattern == 'rand':
                mask = build_rand_mask(gtruth.shape, args.samp)                 # construct a random pattern
            if args.pattern == 'grid':
                mask = build_grid_mask(gtruth.shape, args.samp)                 # construct a grid pattern
            sparse = np.where(mask != 0, gtruth, 0.0)                           # sample with pattern
            if limit_depth:
                sparse[gtruth > (depth_thr * 256.0)] = 0.0                      # remove samples above depth_thr (if limit_depth == True)
            imageio.imwrite(sparse_savepath, sparse.astype('uint16'))           # save the result
