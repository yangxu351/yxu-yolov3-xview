import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    
    path = '/data/users/yang/code/yxu-yolov3-xview/trn_patch_images/'
    
    a = np.load(os.path.join(path, 'train_batch_syn_RC4_v95_epoch0_batch0.npy'))
    a6 = np.load(os.path.join(path, 'train_batch_syn_RC4_v96_epoch0_batch0.npy'))
    a7 = np.load(os.path.join(path, 'train_batch_syn_RC4_v97_epoch0_batch0.npy'))
    a8 = np.load(os.path.join(path, 'train_batch_syn_RC4_v98_epoch0_batch0.npy'))
    a9 = np.load(os.path.join(path, 'train_batch_syn_RC4_v99_epoch0_batch0.npy'))
    
    batch_size = a.shape[0]
    
    for i in range(batch_size):
        plt.imshow((a[i, :, :, :] - a6[i, :, :, :])*255)
    
    plt.close()
    