from PIL import ImageColor
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    ## rc1 (137, 116, 97) (22.1, 19.5, 17.1)
    # hexes = "#685C45;#786755;#897954;#725043;#7F7C5D;#8D7C6A;#A68463;#A48273;#A48273;#987C66;#B39582;#655055;#858A78;#7B6A59;#8A6759;#857861"
    # num_aft = 450*50
    # base_ix = 32
    # rare_cls = 1
    # half = True
    ## rc2
    # # hexes = '#B4A7A4;#C4BCB9;#B6AEA8;' # wing
    # hexes = '#E5DAD8;#F5EAE1;#E6E5E6;#EBE6E1;#DAD9DE;#FFFEFF;#FEF9F7' # body
    # num_aft = 450*22
    # base_ix = 12
    # rare_cls = 2
    # half = False
    ## rc3
    # hexes = '#FFFFF9;#FFFFEF;#FFFFFB;#F9F5E6;#F7F6F0;#EFEEE7'
    # num_aft = 450*34
    # base_ix = 12
    # rare_cls = 3
    # half = False
    ## rc4
    # hexes = '#4C4B4E;#454545;#4A4744;#3D382C;#49483B;#453F37;#403B31;#504E41'
    # num_aft = 450*28
    # base_ix = 23
    # rare_cls = 4
    # half = True
    ## rc5
    hexes = '#C28634;#9C6932;#AA752B;#D89B43;#CD9644;#846830'
    num_aft = 450*50
    base_ix = 12
    rare_cls = 5
    half = True

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
    print('rgb mean', (r_mean, g_mean, b_mean))
    print('rgb std', (r_std, g_std, b_std))
    num_hex = len(hex_list)
    r_std_list = []
    g_std_list = []
    b_std_list = []
    r_mean_list = []
    g_mean_list = []
    b_mean_list = []
    for color_pro in range(11):
        if half:
            bias = color_pro*25.5/2
        else:
            bias = color_pro*25.5
        ix = color_pro + base_ix
        r_unif_list = []
        g_unif_list = []
        b_unif_list = []
        for i in range(num_hex):
            if half:
                r_unif = np.random.uniform(r_list[i] - bias, r_list[i] + bias, num_aft // num_hex)
                g_unif = np.random.uniform(g_list[i] - bias, g_list[i] + bias, num_aft // num_hex)
                b_unif = np.random.uniform(b_list[i] - bias, b_list[i] + bias, num_aft // num_hex)
            else:
                r_unif = np.random.uniform(r_list[i] - bias, r_list[i], num_aft // num_hex)
                g_unif = np.random.uniform(g_list[i] - bias, g_list[i], num_aft // num_hex)
                b_unif = np.random.uniform(b_list[i] - bias, b_list[i], num_aft // num_hex)
            print(r_unif)
            r_unif_list.extend(r_unif)
            g_unif_list.extend(g_unif)
            b_unif_list.extend(b_unif)
        print('mean, std r', np.mean(r_unif_list), np.std(r_unif_list))
        print('mean, std g', np.mean(g_unif_list), np.std(g_unif_list))
        print('mean, std b', np.mean(b_unif_list), np.std(b_unif_list))
        r_mean_list.append(np.mean(r_unif_list))
        g_mean_list.append(np.mean(g_unif_list))
        b_mean_list.append(np.mean(b_unif_list))
        r_std_list.append(np.std(r_unif_list))
        g_std_list.append(np.std(g_unif_list))
        b_std_list.append(np.std(b_unif_list))
    print('r std', r_std_list)
    print('g std', g_std_list)
    print('b std', b_std_list)
    rgb_mean = np.array([s for s in zip(r_mean_list, g_mean_list, b_mean_list)], dtype=np.int)
    rgb_std = np.around([s for s in zip(r_std_list, g_std_list, b_std_list)], decimals=1)
    print('rgb_mean', rgb_mean)
    print('rgb_std', rgb_std)

    df_rgb = pd.DataFrame(columns=['Version', 'rgb_mean', 'rgb_std'], index=None)
    for ix in range(11):
        vix = ix + base_ix
        df_rgb = df_rgb.append({'Version':vix, 'rgb_mean':rgb_mean[ix], 'rgb_std':rgb_std[ix]}, ignore_index=True)
    print(df_rgb)
    if rare_cls == 1:
        mode = 'w'
    else:
        mode = 'a'
    with pd.ExcelWriter(os.path.join('/media/lab/Yang/code/yolov3/result_output/1_cls', 'rare_class_color_bias.xlsx'), mode=mode) as writer:
        df_rgb.to_excel(writer, sheet_name='rc{}'.format(rare_cls), index=False)


