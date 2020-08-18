from PIL import ImageColor
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    np.random.seed(1)

    ## rc1 (137, 116, 97) (22.1, 19.5, 17.1)
    left_bias = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    right_bias = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    base_version = 50
    hexes = "#685C45;#786755;#897954;#8D7C6A;#7B6A59;#857861"
    rare_cls = 1

    ## rc2
    # left_bias = [0.5, -0.5, -1.5, -2.5, -3.5, -4.5]
    # right_bias = [-0.5, -1.5, -2.5, -3.5, -4.5, -5.5]
    # hexes = '#E5DAD8;#F5EAE1;#E6E5E6;#EBE6E1;#FFFEFF;#FEF9F7' # body
    # num_img = 450
    # base_ix = 30
    # rare_cls = 2

    ## rc3
    # left_bias = [0, -1.5, -2.5, -3.5, -4.5, -5.5]
    # right_bias = [0, -0.5, -1.5, -2.5, -3.5, -4.5]
    # hexes = '#FFFFF9;#FFFFEF;#FFFFFB;#F9F5E6;#F7F6F0;#EFEEE7'
    # num_img = 450
    # base_ix = 30
    # rare_cls = 3

    ## rc4
    # left_bias = [-1.5, 0, 0.5, 1.5, 2.5, 3.5]
    # right_bias = [-0.5, 0, 1.5, 2.5, 3.5, 4.5]
    # hexes = '#4C4B4E;#454545;#4A4744;#3D382C;#49483B;#453F37;#403B31;#504E41'
    # num_aft = 450
    # base_ix = 30
    # rare_cls = 4

    ## rc5
    # hexes = '#C28634;#9C6932;#AA752B;#D89B43;#CD9644;#846830'
    # left_bias = [-2.5, -1.5, 0, 0.5, 1.5, 2.5]
    # right_bias = [-1.5, -0.5, 0, 1.5, 2.5, 3.5]
    # num_aft = 450
    # base_ix = 20
    # rare_cls = 5


    # seeds = np.random.choice(range(5000), 450)
    hex_list = [s for s in hexes.split(';')]
    r_list = []
    g_list = []
    b_list = []
    for hex in hex_list:
        (r, g, b) = ImageColor.getcolor(hex, "RGB")
        r_list.append(r)
        g_list.append(g)
        b_list.append(b)
    r_mean = np.round(np.mean(r_list)).astype(np.int)
    g_mean = np.round(np.mean(g_list)).astype(np.int)
    b_mean = np.round(np.mean(b_list)).astype(np.int)
    r_std = np.std(r_list)
    g_std = np.std(g_list)
    b_std = np.std(b_list)
    rgb_mean = np.array([r_mean, g_mean, b_mean])
    rgb_std = np.around([r_std, g_std, b_std], decimals=1)
    print('rgb mean', rgb_mean)
    print('rgb std', rgb_std)

    for ix in range(len(left_bias)):
        lbp = left_bias[ix] * 25.5
        rbp = right_bias[ix] * 25.5
        print('lbp, rbp', lbp, rbp)
        low = np.clip(rgb_mean + lbp, 0, 255)
        high = np.clip(rgb_mean + rbp, 0, 255)
        # print('low', low, 'high', high)
        unif_rgb_mean = (low+high)/2
        unif_rgb_std = np.power(high-low,2)/12
        print('unif mean, std', unif_rgb_mean, unif_rgb_std)

    df_rgb = pd.DataFrame(columns=['Version', 'rgb_mean', 'rgb_std'], index=None)
    for ix in range(len(left_bias)): # range(11)
        vix = ix + base_version
        df_rgb = df_rgb.append({'Version':vix, 'rgb_mean':rgb_mean[ix], 'rgb_std':rgb_std[ix]}, ignore_index=True)
    print(df_rgb)
    if rare_cls == 1:
        mode = 'w'
    else:
        mode = 'a'
    with pd.ExcelWriter(os.path.join('/media/lab/Yang/code/yolov3/result_output/1_cls', 'rare_class_color_bias.xlsx'), mode=mode) as writer:
        df_rgb.to_excel(writer, sheet_name='rc{}'.format(rare_cls), index=False)



