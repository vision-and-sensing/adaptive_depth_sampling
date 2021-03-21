from shutil import copyfile
import os
import argparse
# copy (and rename) generated ImpMaps to dataset dir

# parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='', help='dataset root path')                           # <----- INSERT DEFAULT VALUE (dataset root dir)
parser.add_argument('--pix2pixhd_path', type=str, default='', help='pix2pixhd root path')                       # <----- INSERT DEFAULT VALUE (pix2pixhd root dir)
parser.add_argument('--net_name', type=str, default='', help='pix2pixhd model name (as named in training)')     # <----- INSERT DEFAULT VALUE (pix2pixhd model name)
parser.add_argument('--generated_set', type=str, default='train', help='generated set (train / validation / test)')
parser.add_argument('--preproc', type=str, default='0.25pow', help='prefix of ImpMap dir as indicator for preprocessing that was used for pix2pixhd training')
args = parser.parse_args()

from_fold = args.pix2pixhd_path + "/results/" + args.net_name + "/test_latest/images/"                      # construct path to generated images dir
to_fold = args.dataset_path + args.generated_set + "/ImpMaps/" + args.preproc + "_" + args.net_name + "/"   # construct path to target dir in dataset
if not os.path.exists(to_fold):
    os.mkdir(to_fold)

image = [img for img in os.listdir(from_fold)]                  # construct list of images
image.sort()

for inx in range(len(image)):                                   # for each image
    if 'synthesized' in image[inx]:                             # of its a synthesized (generated) image that we would like to copy
        imname = image[inx].split('_')[0] + '.png'              # change name (to fit naming in dataset)
        copyfile(from_fold + image[inx], to_fold+imname)        # copy file
