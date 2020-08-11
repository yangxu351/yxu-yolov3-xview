import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    ## rc1
    # rc_low = 12.5
    # rc_high = 14
    # num_aft = 450*50
    # base_ix = 43
    # rare_cls = 1
    # size_range = [x for x in range(-1, 7)]

    ## rc2
    # rc_low = 37
    # rc_high = 40
    # num_aft = 450*22
    # base_ix = 23
    # rare_cls = 2
    # size_range = [x for x in range(-3, 5)]

    ## rc3
    # rc_low = 22
    # rc_high = 28
    # num_aft = 450*34
    # base_ix = 23
    # rare_cls = 3
    # size_range = [x for x in range(-2, 6)]

    ## rc4
    # rc_low = 30
    # rc_high = 32
    # num_aft = 450*28
    # base_ix = 23
    # rare_cls = 4
    # size_range = [x for x in range(-3, 5)]

    ## rc5
    rc_low = 7.5
    rc_high = 10
    num_aft = 441*5
    base_ix = 23
    rare_cls = 5
    size_range = [x for x in range(0, 8)]

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
