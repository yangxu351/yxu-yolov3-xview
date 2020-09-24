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


def generate_normal(sd, mu, sigma, size):
    np.random.seed(sd)
    u_list = np.random.random(size=size)
    v_list = np.random.random(size=size)
    gs = []
    gsstr = ""
    for ix in range(size):
        u = u_list[ix]
        v = v_list[ix]
        # z1 = np.sqrt(-2 * np.log(u)) * np.sin(2 * np.pi * v)
        z2 = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)
        print('u', u, 'v', v, 'z2', z2)
    # x1 = mu + z1 * sigma
        x2 = mu + z2 * sigma
        gs.append(x2)
        gsstr += '{:.2f};'.format(x2)

    return gs, gsstr

def gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft):
    np.random.seed(1)
    mu = np.array(mu)
    save_dir = '../../result_output/RC_size/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # df_size = pd.DataFrame(columns=['Version', 'mean_body', 'mean_wing', 'size_sqsigma'], index=None)
    f_txt = open(os.path.join(save_dir, 'RC{}_size.txt'.format(rare_cls)), 'w')
    gs = []
    for ix, ssig in enumerate(ssig_list):
        diag = ssig * np.diag(mu)
        gs = np.random.multivariate_normal(mu, diag, num_aft)
        # plt.hist(gs, 30)
        # a = np.random.choice(gs[:, 0], 450*2)
        # print("mean, std", np.mean(a), np.std(a))
        # plt.hist(a, 30)
        # plt.show()
        body_str = ''
        wing_str = ''
        for jx in range(gs.shape[0]):
            body_str += '{:.2f};'.format(gs[jx, 0])
            wing_str += '{:.2f};'.format(gs[jx, 1])

        f_txt.write('@Hidden\nattr body{}= "{}"\n\n'.format(int(ssig*100), body_str))
        f_txt.write('@Hidden\nattr wing{}= "{}"\n\n'.format(int(ssig*100), wing_str))
        # df_size = df_size.append({'Version':ix, 'mean_body':mu[0], 'mean_wing':mu[1], 'size_sqsigma':ssig}, ignore_index=True)
    f_txt.close()
    # if rare_cls == 1:
    #     mode = 'w'
    # else:
    #     mode = 'a'
    # with pd.ExcelWriter(os.path.join(save_dir, 'rare_class_size_mean_sigma.xlsx'), mode=mode) as writer:
    #     df_size.to_excel(writer, sheet_name='rc{}'.format(rare_cls), index=False)
    # df_size.tocsv(os.path.join(save_dir, 'RC{}_size_mean_sigma.csv'.format(rare_cls)), index=False)


if __name__ == '__main__':
    ## rc1
    num_aft = int(450*7*3)
    rare_cls = 1
    mu = [13, 7]
    ssig_list = [0, 0.03, 0.06, 0.09, 0.12]
    gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft)

    # # ## rc2
    # num_aft = int(450*7*3)
    # rare_cls = 2
    # mu = [38.3, 33]
    # ssig_list = [0, 0.03, 0.06, 0.09, 0.12]
    # gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft)
    #
    # ## rc3
    # num_aft = int(450*7*3)
    # rare_cls = 3
    # mu = [25.7, 25.5]
    # ssig_list = [0, 0.03, 0.06, 0.09, 0.12]
    # gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft)
    #
    # ## rc4
    # num_aft = int(450*7*3)
    # rare_cls = 4
    # mu = [31, 39]
    # ssig_list = [0, 0.03, 0.06, 0.09, 0.12]
    # gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft)
    #
    # ## rc5
    # num_aft = int(450*7*3)
    # rare_cls = 5
    # mu = [8.3, 19.9]
    # ssig_list = [0, 0.03, 0.06, 0.09, 0.12]
    # gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft)


    # mu = 25.7
    # ssig = mu*0.12
    # print('ssig', ssig)
    # all_gs = []
    # all_gsstr = ""
    # np.random.seed(1)
    # seeds = np.random.choice(5000, 450)
    # for i in range(450):
    #     sd = seeds[i]
    #     gs, gsstr = generate_normal(sd, mu, ssig, size=7*2)
    #     all_gs.extend(gs)
    #     all_gsstr += gsstr
    # print('gs min {:.3f}  max {:.3f}'.format(np.min(all_gs), np.max(all_gs)))
    # print('gs len {} mean {:.3f} std {:.3f}'.format(len(all_gs), np.mean(all_gs), np.std(all_gs)))
    # plt.hist(np.array(gs), bins=10, density=True)
    # plt.show()
