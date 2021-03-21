import matplotlib.pyplot as plt     # for debug
import numpy as np
import scipy.ndimage
import scipy.signal
import skimage
import skimage.io
import skimage.transform
import skimage.feature
import torch
import random
import imageio
import sys
import os
# utils file

# parameters
depth_thr = 100.0   # 100m threshold

# read FusionNet
FusionNetPath = ""        # <----- INSERT DEFAULT VALUE (FusionNet root dir)
sys.path.append(FusionNetPath)
from Models.model import uncertainty_net as FusionNetDef

# a list of bad images (badly rendered) that will be ignored in test (results.py)
bad_imgs = ('5000160.png', '5000161.png', '5000162.png', '5000163.png', '5000164.png', '5000165.png', '5000166.png',
            '5000167.png', '5000168.png', '5000169.png', '5000170.png', '5000171.png', '5000172.png', '5000173.png',
            '5000174.png', '5000175.png', '5000176.png', '5000177.png', '5000178.png', '5000179.png', '5000180.png',
            '5000181.png', '5000182.png', '5000183.png', '5000184.png', '5000185.png', '5000186.png', '5000187.png',
            '5000188.png', '5000189.png', '5000190.png', '5000191.png', '5000192.png', '5000193.png', '5000194.png',
            '5000195.png', '5100196.png', '5100197.png', '5100198.png', '5100199.png', '5100200.png', '5100201.png',
            '5100202.png', '5100203.png', '5100204.png', '5100205.png', '5100206.png', '5100207.png', '5100208.png',
            '5100209.png')


def net_paths(samp, err, sig, grd):              # <----- INSERT DEFAULT VALUE (FusionNet trained models paths)
    # a bank of paths to trained networks (random, grid, superpixels, importance)
    if samp == 0.01:
        if err == 'RMSE':
            rand_model_path = ""
            grid_model_path = ""
            sp_model_path = ""
            if grd == 0.0 and sig == 2.0:
                imp_model_path = ""
            if grd == 0.1 and sig == 2.0:
                imp_model_path = ""
            if grd == 0.05 and sig == 1.0:
                imp_model_path = ""
        if err == 'REL':
            rand_model_path = ""
            grid_model_path = ""
            sp_model_path = ""
            if grd == 0.05 and sig == 3.0:
                imp_model_path = ""
    return rand_model_path, grid_model_path, sp_model_path, imp_model_path


def valid_mask(gt, ignore_mask=None):
    # caluclates mask where 0 < gt < depth_thr and not in ignored mask
    mask = (0 < gt) * (gt <= depth_thr)
    if ignore_mask is not None:
        mask = mask * np.logical_not(ignore_mask)
    return mask


def mse_err(gt, pred, ignore_mask=None):
    # MSE error
    e = gt - pred
    se = np.square(e)
    val_mask = valid_mask(gt, ignore_mask)
    mse = np.mean(se[val_mask])
    return mse


def rmse_err(gt, pred, ignore_mask=None):
    # RMSE error
    e = gt - pred
    se = np.square(e)
    val_mask = valid_mask(gt, ignore_mask)
    mse = np.mean(se[val_mask])
    rmse = np.sqrt(mse)
    return rmse


def psnr_err(gt, pred, ignore_mask=None, r=100.0):
    # PSNR value
    e = gt - pred
    se = np.square(e)
    val_mask = valid_mask(gt, ignore_mask)
    mse = np.mean(se[val_mask])
    psnr = 10 * np.log10((r ** 2)/mse)
    return psnr


def rel_err(gt, pred, ignore_mask=None):
    # REL error
    err = np.abs((pred - gt) / (gt + 1e-10))
    val_mask = valid_mask(gt, ignore_mask)
    rel = np.mean(err[val_mask])
    return rel


def mad_err(gt, pred, ignore_mask=None):
    # MAD error
    d = gt - pred
    ad = np.abs(d)
    val_mask = valid_mask(gt, ignore_mask)
    mad = np.mean(ad[val_mask])
    return mad


def tukey_err(gt, pred, ignore_mask=None, dmel=10.0):
    # Tukey error
    d = gt - pred
    ad = np.abs(d)
    tukeypw = np.where(ad > dmel, 1, 1 - (1 - (ad/dmel) ** 2) ** 3)
    val_mask = valid_mask(gt, ignore_mask)
    tukey = np.mean(tukeypw[val_mask])
    return tukey


def def_error(err_type):
    # error definition based on string err_type
    if err_type == 'MSE':
        err_measure = mse_err
    if err_type == 'RMSE':
        err_measure = rmse_err
    if err_type == 'PSNR':
        err_measure = psnr_err
    if err_type == 'REL':
        err_measure = rel_err
    if err_type == 'MAD':
        err_measure = mad_err
    if err_type == 'TUKEY':
        err_measure = tukey_err
    return err_measure


def fusionnet_def(modelpath):
    # definition of FusionNet based on trained model path
    model = FusionNetDef(in_channels=4, thres=0)
    model = torch.nn.DataParallel(model).cuda()
    state_dict = torch.load(modelpath)
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    return model


def fusionnet_inp(model, imgL, sparse):
    # using RGB image and saprse depth as input to a trained FusionNet model

    sparse[sparse > depth_thr] = 0

    input = np.concatenate((sparse, imgL), axis=1)
    input = torch.FloatTensor(input).cuda()

    pred1, _, _, _ = model(input, 0)

    pred = pred1.cpu().detach().numpy()

    return pred


def makeGaussian(sigm, size):
    # construct a 2D Gaussian
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    x0 = y0 = int(round((size - 1) / 2))
    if sigm != 0:
        kers = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigm) ** 2))
    else:
        kers = np.zeros((size, size))
        kers[x0, y0] = 1.0
    kers = kers.astype('float64')
    return kers


def pick_max(imp_map, samp_perc, sigma=0.0):
    # Gaussian Sampling
    l = 2.5                         # sigmas to each side

    ltes = int(round(l * sigma))    # length to each side
    n = int(round(samp_perc * imp_map.shape[0] * imp_map.shape[1]))

    imp_map_pad = np.pad(imp_map, ltes, mode='constant').astype('float64')

    samp = np.zeros_like(imp_map)

    kers = makeGaussian(sigma, size=(2 * ltes + 1))
    g = 1 - kers
    for s in range(n):

        ind = np.unravel_index(np.argmax(imp_map_pad), imp_map_pad.shape)
        samp[ind[0] - ltes, ind[1] - ltes] = 1.0

        if sigma == 0.0:
            imp_map_pad[ind] = 0
        else:
            imp_map_pad[ind[0] - ltes:ind[0] + ltes + 1, ind[1] - ltes:ind[1] + ltes + 1] *= g
    return samp


def build_rand_mask(res, samp_perc):
    # build a random mask
    tot_pix = res[0] * res[1]
    num_of_samp = int(round(samp_perc * tot_pix))
    arr = np.array([0] * (tot_pix - num_of_samp) + [1] * num_of_samp)
    np.random.shuffle(arr)
    arr = np.reshape(arr, res)
    return arr


def build_grid_mask(res, samp_perc, prop='square'):
    # build a grid mask
    sparse = np.zeros(res)
    x = res[0]
    y = res[1]
    pix_per_unit = 1 / samp_perc

    if prop == 'square':
        d = np.sqrt(pix_per_unit)

        tot_sampy = int(y/d+0.5)
        tot_sampx = int(x/d+0.5)

        st_x = (x - d * (tot_sampx - 1))/2
        st_y = (y - d * (tot_sampy - 1))/2

        for i in range(tot_sampy):
            for j in range(tot_sampx):
                sx = int(round(st_x + d * j))
                sy = int(round(st_y + d * i))
                sparse[sx, sy] = 1.0

    if prop == 'rect':
        dx = np.sqrt(pix_per_unit * x/y)
        dy = dx*y/x

        st_x = (dx / 2) - 1
        st_y = (dy / 2) - 1

        for i in range(int(y/dy+0.5)):
            for j in range(int(x/dx+0.5)):
                sx = int(round(st_x + dx * j))
                sy = int(round(st_y + dy * i))
                sparse[sx, sy] = 1.0
    return sparse


def mark_samp_patterns(img, *patterns):
    # mask sampling patterns on RGB image
    colors = [[0, 0, 255], [0, 255, 0], [0, 0, 0], [255, 0, 0]]     # blue, green, black, red
    img_col = img.copy()
    for p in range(len(patterns)):
        col = colors[p]
        pattern = patterns[p]

        if np.argmin(img.shape) == 0:
            r = img[0, :, :]
            g = img[1, :, :]
            b = img[2, :, :]
        if np.argmin(img.shape) == 2:
            r = img[:, :, 0]
            g = img[:, :, 1]
            b = img[:, :, 2]

        r[pattern > 0] = col[0]
        g[pattern > 0] = col[1]
        b[pattern > 0] = col[2]

        if np.argmin(img.shape) == 0:
            r = np.expand_dims(r, 0)
            g = np.expand_dims(g, 0)
            b = np.expand_dims(b, 0)
            img_col = np.concatenate((r, g, b), 0)
        if np.argmin(img.shape) == 2:
            r = np.expand_dims(r, 2)
            g = np.expand_dims(g, 2)
            b = np.expand_dims(b, 2)
            img_col = np.concatenate((r, g, b), 2)

    return img_col


def mark_samp_patterns2(img, dil_iter=1, *patterns):
    # mask sampling patterns on RGB image
    bl = np.zeros_like(img)
    bl[2, :, :] = 255
    gr = np.zeros_like(img)
    gr[1, :, :] = 255
    rd = np.zeros_like(img)
    rd[0, :, :] = 255
    prpl = np.zeros_like(img)
    prpl[0, :, :] = 128
    prpl[2, :, :] = 255
    colors = (bl, gr, prpl, rd)

    img_col = img.copy()
    for p in range(len(patterns)):
        col = colors[p]
        pattern = patterns[p]
        for i in range(dil_iter):
            pattern = mydil(pattern)
        img_col = np.where(pattern > 0, col, img_col)
    return img_col


def rand_mask_max(w, samp_perc):
    # create a random mask in binary region w
    num_of_samp = int(round(w.shape[0] * w.shape[1] * samp_perc))
    samp = np.zeros_like(w)
    if np.count_nonzero(w == w.max()) < num_of_samp:
        ind = np.random.choice(np.flatnonzero(w == w.max()), size=np.count_nonzero(w == w.max()), replace=False)
        ind = np.unravel_index(ind, w.shape)
        samp[ind] = 1
        ind2 = np.random.choice(np.flatnonzero(w - 2 * samp == w.max() - 1), size=num_of_samp - np.count_nonzero(w == w.max()), replace=False)
        ind2 = np.unravel_index(ind2, w.shape)
        samp[ind2] = 1
    else:
        ind = np.random.choice(np.flatnonzero(w == w.max()), size=num_of_samp, replace=False)
        ind = np.unravel_index(ind, w.shape)
        samp[ind] = 1
    assert np.count_nonzero(samp == 1) == num_of_samp
    assert np.count_nonzero(samp == 0) == w.shape[0] * w.shape[1] - num_of_samp
    new_w = w - samp
    if new_w.max() == 1 and np.count_nonzero(new_w == 1) < num_of_samp:
        new_w[new_w == 1] = 0
    if new_w.max() == 1:
        debugg_var = 1
    return samp, new_w


def mydil(im1):
    # binary dilation
    im2 = im1.copy()
    struct1 = scipy.ndimage.generate_binary_structure(2, 2)
    im2 = scipy.ndimage.binary_dilation(im2, structure=struct1).astype(im2.dtype)
    return im2


def mydil_sparse(im1, fil_val=0):
    # non-binary dilation (for sparse depth)
    im2 = im1.copy()
    invalid = im2 == fil_val
    valid = im2 != fil_val
    ind = scipy.ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    im2 = im2[tuple(ind)]
    im2 = im2 * mydil(valid)
    return im2


def save_img_cm(img_path, cmap='viridis', mind=0.0, maxd=20.0, save_dir='/home/tcenov/Depth/saves/'):
    # convert image to CM and save
    cm = plt.get_cmap(cmap)
    img_name = img_path.split('/')[-1]
    img = skimage.io.imread(img_path, as_gray=True) / 256.0
    img = np.clip(img, mind, maxd) / maxd
    img = cm(img)
    skimage.io.imsave(save_dir + img_name, img)


def collect_paths(f1, f2, f3, seq, subseq):
    # construct paths
    fold1 = [img for img in os.listdir(f1)]
    fold2 = [img for img in os.listdir(f2)]
    fold3 = [img for img in os.listdir(f3)]

    fold1imgs = [f1 + img for img in fold1]
    fold2imgs = [f2 + img for img in fold2]
    fold3imgs = [f3 + img for img in fold3]

    imgs = fold1 + fold2 + fold3
    fullimgs = fold1imgs + fold2imgs + fold3imgs

    sortedpaths = [x for _, x in sorted(zip(imgs, fullimgs))]
    sortedpaths[:] = [tup for tup in sortedpaths if tup[-11] == seq and tup[-10] == subseq]
    return sortedpaths


def plot_res(gtruth, imgL, rand_pred, grid_pred, sp_pred, imp_pred, maxd=30, dt=depth_thr):
    # plot results
    rand_pred = rand_pred[0, 0]
    grid_pred = grid_pred[0, 0]
    sp_pred = sp_pred[0, 0]
    imp_pred = imp_pred[0, 0]

    invalid = gtruth > dt
    rand_pred[invalid] = gtruth[invalid]
    grid_pred[invalid] = gtruth[invalid]
    sp_pred[invalid] = gtruth[invalid]
    imp_pred[invalid] = gtruth[invalid]

    plt.figure()
    plt.imshow(np.transpose(imgL[0], (1, 2, 0)))
    plt.title('image')

    plt.figure()
    plt.imshow(gtruth, vmin=0, vmax=maxd)
    plt.title('gtruth')

    plt.figure()
    plt.imshow(rand_pred, vmin=0, vmax=maxd)
    plt.title('rand pred')

    plt.figure()
    plt.imshow(grid_pred, vmin=0, vmax=maxd)
    plt.title('grid pred')

    plt.figure()
    plt.imshow(sp_pred, vmin=0, vmax=maxd)
    plt.title('sp pred')

    plt.figure()
    plt.imshow(imp_pred, vmin=0, vmax=maxd)
    plt.title('imp pred')


def save_res_imgs(gtruth, imgL, rand_pred, grid_pred, sp_pred, imp_pred, imp_sparse_path, impr,
                  rand_mask, grid_mask, sp_mask, imp_mask,
                  rand_sparse, grid_sparse, sp_sparse, imp_sparse,
                  err_type, rand_err, grid_err, sp_err, imp_err, imname, ind,
                  mind=0, maxd=30, dt=depth_thr, cmap='viridis', save_root="/home/"):
    # save results as images
    save_path = os.path.join(save_root, "results_" + err_type + "_NEW")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if impr > 0:
        impr_str = 'G' + str(round(impr * 100, 2))
    else:
        impr_str = 'B' + str(round(-1.0 * impr * 100, 2))

    fold_name = imname[:-4] + '_ind'+str(ind) + '_' + impr_str
    save_fold_path = os.path.join(save_path, fold_name)
    if not os.path.exists(save_fold_path):
        os.mkdir(save_fold_path)

    impmap_type = imp_sparse_path.split('/')[-2].split('unnorm_')[1].split('_grd')[0]
    test_path = imp_sparse_path.split('LiDAR')[0]
    impmap_fold_path = os.path.join(test_path, 'ImpMaps', impmap_type)
    impmap = skimage.io.imread(os.path.join(impmap_fold_path, imname)).astype('float64')

    impr_rand = np.round((rand_err[0] - imp_err) / rand_err[0] * 100, 2)
    impr_grid = np.round((grid_err - imp_err) / grid_err * 100, 2)
    impr_sp = np.round((sp_err - imp_err) / sp_err * 100, 2)

    f = open(os.path.join(save_fold_path, 'stats.txt'), "w+")
    f.write("Image name: %s\r\n" % imname)
    f.write("Image index: %d\r\n\n" % ind)
    f.write("Sampling pattern | %s\r\n" % err_type)
    f.write("Random           | %.4f\r\n" % rand_err[0])
    f.write("Grid             | %.4f\r\n" % grid_err)
    f.write("SuperPixels      | %.4f\r\n" % sp_err)
    f.write("Importance       | %.4f\r\n\n" % imp_err)
    f.write("Improvement over random [%%]: %.2f\r\n" % impr_rand)
    f.write("Improvement over grid [%%]  : %.2f\r\n" % impr_grid)
    f.write("Improvement over sp [%%]    : %.2f\r\n" % impr_sp)
    f.close()

    cm = plt.get_cmap(cmap)

    imgL = imgL[0]
    rand_pred = rand_pred[0, 0]
    grid_pred = grid_pred[0, 0]
    sp_pred = sp_pred[0, 0]
    imp_pred = imp_pred[0, 0]

    imgL_marked0 = mark_samp_patterns2(imgL, 0, rand_mask, grid_mask, sp_mask, imp_mask)
    imgL_marked1 = mark_samp_patterns2(imgL, 1, rand_mask, grid_mask, sp_mask, imp_mask)

    invalid = gtruth > dt
    rand_pred[invalid] = gtruth[invalid]
    grid_pred[invalid] = gtruth[invalid]
    sp_pred[invalid] = gtruth[invalid]
    imp_pred[invalid] = gtruth[invalid]

    rand_dif = np.abs(gtruth - rand_pred)
    grid_dif = np.abs(gtruth - grid_pred)
    sp_dif = np.abs(gtruth - sp_pred)
    imp_dif = np.abs(gtruth - imp_pred)
    maxdif = np.max((rand_dif, grid_dif, sp_dif, imp_dif))

    #impmap = np.sqrt(impmap)
    impmap_cm = cm(impmap / impmap.max())

    rand_dif = cm(rand_dif / maxdif)
    grid_dif = cm(grid_dif / maxdif)
    sp_dif = cm(sp_dif / maxdif)
    imp_dif = cm(imp_dif / maxdif)

    rand_mask = cm(mydil(rand_mask * 1.0))
    grid_mask = cm(mydil(grid_mask * 1.0))
    sp_mask = cm(mydil(sp_mask * 1.0))
    imp_mask = cm(mydil(imp_mask * 1.0))

    rand_sparse = cm(np.clip(mydil_sparse(rand_sparse[0, 0]), mind, maxd) / maxd)
    grid_sparse = cm(np.clip(mydil_sparse(grid_sparse[0, 0]), mind, maxd) / maxd)
    sp_sparse = cm(np.clip(mydil_sparse(sp_sparse[0, 0]), mind, maxd) / maxd)
    imp_sparse = cm(np.clip(mydil_sparse(imp_sparse[0, 0]), mind, maxd) / maxd)

    gtruth_cm = cm(np.clip(gtruth, mind, maxd) / maxd)
    rand_pred_cm = cm(np.clip(rand_pred, mind, maxd) / maxd)
    grid_pred_cm = cm(np.clip(grid_pred, mind, maxd) / maxd)
    sp_pred_cm = cm(np.clip(sp_pred, mind, maxd) / maxd)
    imp_pred_cm = cm(np.clip(imp_pred, mind, maxd) / maxd)

    skimage.io.imsave(os.path.join(save_fold_path, 'RGB.png'), (np.transpose(imgL, (1, 2, 0))*255.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'RGB_marked0.png'), (np.transpose(imgL_marked0, (1, 2, 0))*255.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'RGB_marked1.png'), (np.transpose(imgL_marked1, (1, 2, 0))*255.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'GT_cm.png'), (gtruth_cm[:, :, :3]*65535.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'GT.png'), (gtruth*255.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'rand_pred_cm.png'), (rand_pred_cm[:, :, :3]*65535.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'rand_pred.png'), (rand_pred*255.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'rand_dif.png'), (rand_dif*65535.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'grid_pred_cm.png'), (grid_pred_cm[:, :, :3]*65535.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'grid_pred.png'), (grid_pred*255.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'grid_dif.png'), (grid_dif[:, :, :3]*65535.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'sp_pred_cm.png'), (sp_pred_cm[:, :, :3]*65535.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'sp_pred.png'), (sp_pred*255.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'sp_dif.png'), (sp_dif[:, :, :3]*65535.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'imp_pred_cm.png'), (imp_pred_cm[:, :, :3]*65535.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'imp_pred.png'), (imp_pred*255.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'imp_dif.png'), (imp_dif[:, :, :3]*65535.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'impmap_cm.png'), (impmap_cm[:, :, :3]*65535.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'impmap.png'), impmap.astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'rand_sparse.png'), (rand_sparse[:, :, :3]*65535.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'grid_sparse.png'), (grid_sparse[:, :, :3]*65535.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'sp_sparse.png'), (sp_sparse[:, :, :3]*65535.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'imp_sparse.png'), (imp_sparse[:, :, :3]*65535.0).astype('uint16'))
    skimage.io.imsave(os.path.join(save_fold_path, 'rand_mask.png'), rand_mask)
    skimage.io.imsave(os.path.join(save_fold_path, 'grid_mask.png'), grid_mask)
    skimage.io.imsave(os.path.join(save_fold_path, 'sp_mask.png'), sp_mask)
    skimage.io.imsave(os.path.join(save_fold_path, 'imp_mask.png'), imp_mask)
