import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def compute_size():
    mean_list = []
    std_list = []
    step = 5
    for size_pro in size_range:
        if size_pro < 0:
            low = rc_low + size_pro*step
            high = rc_high + size_pro*step
        else:
            low = rc_low + size_pro*step
            high = rc_high + size_pro*step
        size_unif = np.random.uniform(low, high, num_aft)
        mean_size = np.mean(size_unif)
        std_size = np.std(size_unif)
        mean_list.append(mean_size)
        std_list.append(std_size)
    mean_list = np.around(mean_list, decimals=3)
    std_list = np.around(std_list, decimals=3)
    print('mean_list ', mean_list)
    print('std list', std_list)
    df_rgb = pd.DataFrame(columns=['Version', 'size_mean', 'size_std'], index=None)

    for ix in range(8):
        vix = ix + base_ix
        df_rgb = df_rgb.append({'Version':vix, 'size_mean':mean_list[ix], 'size_std':std_list[ix]}, ignore_index=True)
    print(df_rgb)
    if rare_cls == 1:
        mode = 'w'
    else:
        mode = 'a'
    with pd.ExcelWriter(os.path.join('/media/lab/Yang/code/yolov3/result_output/1_cls', 'rare_class_size_bias.xlsx'), mode=mode) as writer:
        df_rgb.to_excel(writer, sheet_name='rc{}'.format(rare_cls), index=False)


def generate_normal(mu, sigma, size):
    np.random.seed(1)
    u = np.random.random(size=size)
    v_list = np.random.random(size=size)
    gs = []
    for v in v_list:
    # z1 = np.sqrt(-2 * np.log(u)) * np.sin(2 * np.pi * v)
        z2 = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)

    # x1 = mu + z1 * sigma
        x2 = mu + z2 * sigma
        gs.append(x2)

    return gs

def gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft):
    np.random.seed(1)
    mu = np.array(mu)
    save_dir = '/media/lab/Yang/code/yolov3/result_output/1_cls/RC_size/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    df_size = pd.DataFrame(columns=['Version', 'mean_body', 'mean_wing', 'size_sqsigma'], index=None)
    f_txt = open(os.path.join(save_dir, 'RC{}_size.txt'.format(rare_cls)), 'w')
    for ix, ssig in enumerate(ssig_list):
        diag = np.diag([ssig, ssig])
        # gs = generate_normal(mu[0], ssig, size=200)
        gs = np.random.multivariate_normal(mu, diag, num_aft)
        # plt.hist(gs, 30)
        # plt.show()
        body_str = ''
        wing_str = ''
        for jx in range(gs.shape[0]):
            body_str += '{:.3f};'.format(gs[jx, 0])
            wing_str += '{:.3f};'.format(gs[jx, 1])

        f_txt.write('body{}: "{}"\n\n'.format(ix, body_str))
        f_txt.write('wing{}: "{}"\n\n'.format(ix, wing_str))
        df_size = df_size.append({'Version':ix, 'mean_body':mu[0], 'mean_wing':mu[1], 'size_sqsigma':ssig}, ignore_index=True)
    f_txt.close()
    if rare_cls == 1:
        mode = 'w'
    else:
        mode = 'a'
    with pd.ExcelWriter(os.path.join(save_dir, 'rare_class_size_mean_sigma.xlsx'), mode=mode) as writer:
        df_size.to_excel(writer, sheet_name='rc{}'.format(rare_cls), index=False)


if __name__ == '__main__':
    ## rc1
    # base_ix = 43
    # size_range = [x for x in range(-1, 7)]
    num_aft = 450*6*3
    rare_cls = 1
    mu = [13, 7]
    ssig_list = [0, 0.03, 0.06, 0.09, 0.12]
    gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft)

    #
    # ## rc2
    # # base_ix = 23
    # # size_range = [x for x in range(-3, 5)]
    # num_aft = 450*6*3
    # rare_cls = 2
    # mu = [38.3, 33]
    # ssig_list = [0, 0.03, 0.06, 0.09, 0.12]
    # gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft)
    #
    # ## rc3
    # # rc_low = 22
    # # rc_high = 28
    # # base_ix = 23
    # # size_range = [x for x in range(-2, 6)]
    # num_aft = 450*6*3
    # rare_cls = 3
    # mu = [25.7, 25.5]
    # ssig_list = [0, 0.03, 0.06, 0.09, 0.12]
    # gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft)
    #
    # ## rc4
    # # rc_low = 30
    # # rc_high = 32
    # # base_ix = 23
    # # size_range = [x for x in range(-3, 5)]
    # num_aft = 450*6*4
    # rare_cls = 4
    # mu = [31, 39]
    # ssig_list = [0, 0.03, 0.06, 0.09, 0.12]
    # gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft)
    #
    # ## rc5
    # # rc_low = 7.5
    # # rc_high = 10
    # # base_ix = 23
    # # size_range = [x for x in range(0, 8)]
    # num_aft = 450*6*3
    # rare_cls = 5
    # mu = [8.3, 19.9]
    # ssig_list = [0, 0.03, 0.06, 0.09, 0.12]
    # gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft)


