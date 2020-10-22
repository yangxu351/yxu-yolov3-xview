import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':

    path = '/media/lab/Yang/code/yolov3/trn_patch_images/syn_RC4/'
    save_path = os.path.join(path, 'diff')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # vers = 96
    # seed0=2
    # epoch0 = 5
    # bt0 = 0
    # i0 = 7
    # seed0=2
    # epoch0 = 1
    # bt0 = 0
    # i0 = 7
    # seed0=0
    # epoch0 = 0
    # bt0 = 0
    # i0 = 2
    # seed0=0
    # epoch0 = 0
    # bt0 = 0
    # i0 = 0
    vers = 96
    seed0=0
    epoch0 = 0
    bt0 = 0
    i0 = 0
    a0 = np.load(os.path.join(path, 'train_batch_syn_RC4_v{}_seed{}_epoch{}_batch{}.npy'.format(vers, seed0, epoch0, bt0))).transpose([0, 2, 3, 1])
    # vers = 104
    # seed0=0
    # epoch0 = 0
    # bt0 = 0
    # i0 = 2
    # vers = 104
    # seed0=0
    # epoch0 = 0
    # bt0 = 0
    # i0 = 2
    # a0 = np.load(os.path.join(path, 'train_batch_syn_RC4_v{}_seed{}_epoch{}_batch{}.npy'.format(vers, seed0, epoch0, bt0))).transpose([0, 2, 3, 1])

    # seed1=2
    # epoch1 = 6
    # bt1 = 1
    # i1 = 4
    # seed1=2
    # epoch1 = 2
    # bt1 = 0
    # i1 = 7
    # seed1=1
    # epoch1 = 0
    # bt1 = 0
    # i1 = 0
    # seed1=2
    # epoch1 = 0
    # bt1 = 0
    # i1 = 3
    # seed1=0
    # epoch1 = 1
    # bt1 = 0
    # i1 = 0
    # a1 = np.load(os.path.join(path, 'train_batch_syn_RC4_v{}_seed{}_epoch{}_batch{}.npy'.format(vers, seed1, epoch1, bt1))).transpose([0, 2, 3, 1])

    # vers = 104
    # seed1=1
    # epoch1 = 0
    # bt1 = 0
    # i1 = 0
    # vers = 104
    # seed1=2
    # epoch1 = 0
    # bt1 = 0
    # i1 = 3
    # a1 = np.load(os.path.join(path, 'train_batch_syn_RC4_v{}_seed{}_epoch{}_batch{}.npy'.format(vers, seed1, epoch1, bt1))).transpose([0, 2, 3, 1])

    fig, axs = plt.subplots(1, 3, figsize=(10, 8))
    axs[0].imshow(a0[i0, :, :, :], interpolation='nearest')
    axs[0].set_title('seed{}_epoch{}_batch{}_i{}'.format(seed0, epoch0, bt0, i0))
    axs[1].imshow(a1[i1, :, :, :], interpolation='nearest')
    axs[1].set_title('seed{}_epoch{}_batch{}_i{}'.format(seed1, epoch1, bt1, i1))
    axs[2].imshow(a0[i0, :, :, :] - a1[i1, :, :, :], interpolation='nearest')
    axs[2].set_title('left-right')
    plt.show()
    plt.savefig(os.path.join(save_path, 'diff_RC4_v{}_seed{}-{}_epoch{}-{}_batch{}-{}_i{}-{}.png'.format(vers, seed0, seed1, epoch0, epoch1,
                                                                                                         bt0, bt1, i0, i1)))
    plt.close()
