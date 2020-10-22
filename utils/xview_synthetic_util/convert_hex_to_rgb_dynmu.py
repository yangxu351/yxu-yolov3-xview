from PIL import ImageColor
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def gaussian_disribution_of_color(body_mu, ssig_list, rare_cls, num_aft, wing_mu=None, cmt='RC'):
    np.random.seed(1)
    body_mu = np.array(body_mu)
    # save_dir = '../../result_output/RC_color/'
    save_dir = '../../data_xview/1_cls/px23whr3_seed17/{}/{}_color/'.format(cmt, cmt)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if wing_mu:
        wing_txt = open(os.path.join(save_dir, '{}{}_wing_color.txt'.format(cmt, rare_cls)), 'w')
    body_txt = open(os.path.join(save_dir, '{}{}_body_color.txt'.format(cmt, rare_cls)), 'w')
    for ix, ssig in enumerate(ssig_list):
        diag_body = ssig * np.diag([1, 1, 1])
        # diag_body = ssig * np.diag([1, 1, 1])
        body_gs = np.random.multivariate_normal(body_mu, diag_body, num_aft) # convoriance
        # plt.hist(body_gs, 30)
        # plt.show()
        body_b = ''
        body_g = ''
        body_r = ''
        # body_gs = np.clip(body_gs, 0, 255)
        for jx in range(body_gs.shape[0]):
            body_b += '{:.1f};'.format(body_gs[jx, 0])
            body_g += '{:.1f};'.format(body_gs[jx, 1])
            body_r += '{:.1f};'.format(body_gs[jx, 2])
        body_txt.write('@Hidden\nattr body_r{}= "{}"\n'.format(ssig, body_b))
        body_txt.write('@Hidden\nattr body_g{}= "{}"\n'.format(ssig, body_g))
        body_txt.write('@Hidden\nattr body_b{}= "{}"\n\n'.format(ssig, body_r))

        if wing_mu:
            diag_wing = ssig * np.diag([1, 1, 1])
            wing_gs = np.random.multivariate_normal(wing_mu, diag_wing, num_aft)
            wing_b = ''
            wing_g = ''
            wing_r = ''
            for jx in range(wing_gs.shape[0]):
                wing_b += '{:.1f};'.format(wing_gs[jx, 0])
                wing_g += '{:.1f};'.format(wing_gs[jx, 1])
                wing_r += '{:.1f};'.format(wing_gs[jx, 2])

            wing_txt.write('@Hidden\nattr wing_b{}= "{}"\n'.format(ssig, wing_r))
            wing_txt.write('@Hidden\nattr wing_g{}= "{}"\n'.format(ssig, wing_g))
            wing_txt.write('@Hidden\nattr wing_r{}= "{}"\n\n'.format(ssig, wing_b))
    body_txt.close()
    if wing_mu:
        wing_txt.close()


def compute_rgb_by_hex(hex_list):
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
    r_std = np.around(np.std(r_list), decimals=2)
    g_std = np.around(np.std(g_list), decimals=2)
    b_std = np.around(np.std(b_list), decimals=2)
    rgb_mean = np.array([r_mean, g_mean, b_mean])
    rgb_std = np.array([r_std, g_std, b_std])
    print('mean', rgb_mean)
    print('std', rgb_std)
    return rgb_mean, rgb_std


def gaussian_disribution_of_promu_rgb(body_mu, ssig_list, rare_cls, num_aft, wing_mu=None, wing_ssig_list=None, cmt='CC'):
    np.random.seed(1)
    body_mu = np.array(body_mu)
    # save_dir = '../../result_output/RC_color/'
    save_dir = '../../data_xview/1_cls/px23whr3_seed17/{}/{}_color/'.format(cmt, cmt)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print(save_dir)
    if wing_mu is not None:
        wing_txt = open(os.path.join(save_dir, '{}{}_wing_color.txt'.format(cmt, rare_cls)), 'w')
    body_txt = open(os.path.join(save_dir, '{}{}_body_color.txt'.format(cmt, rare_cls)), 'w')
    for ix, ssig in enumerate(ssig_list):
        diag_body = np.diag(np.power(ssig*body_mu, 2))
        body_gs = np.random.multivariate_normal(body_mu, diag_body, num_aft) # convoriance
        body_b = ''
        body_g = ''
        body_r = ''
        # body_gs = np.clip(body_gs, 0, 255)
        for jx in range(body_gs.shape[0]):
            body_b += '{:.1f};'.format(body_gs[jx, 0])
            body_g += '{:.1f};'.format(body_gs[jx, 1])
            body_r += '{:.1f};'.format(body_gs[jx, 2])
        body_txt.write('@Hidden\nattr body_r{:g}= "{}"\n'.format(ssig[0]*100, body_b))
        body_txt.write('@Hidden\nattr body_g{:g}= "{}"\n'.format(ssig[1]*100, body_g))
        body_txt.write('@Hidden\nattr body_b{:g}= "{}"\n\n'.format(ssig[2]*100, body_r))
    body_txt.close()
    if wing_mu is not None:
        for ix, ssig in enumerate(wing_ssig_list):
            diag_wing = np.diag(np.power(ssig*wing_mu, 2))
            wing_gs = np.random.multivariate_normal(wing_mu, diag_wing, num_aft)
            wing_b = ''
            wing_g = ''
            wing_r = ''
            for jx in range(wing_gs.shape[0]):
                wing_b += '{:.1f};'.format(wing_gs[jx, 0])
                wing_g += '{:.1f};'.format(wing_gs[jx, 1])
                wing_r += '{:.1f};'.format(wing_gs[jx, 2])

            wing_txt.write('@Hidden\nattr wing_r{:g}= "{}"\n'.format(ssig[0]*100, wing_r))
            wing_txt.write('@Hidden\nattr wing_g{:g}= "{}"\n'.format(ssig[1]*100, wing_g))
            wing_txt.write('@Hidden\nattr wing_b{:g}= "{}"\n\n'.format(ssig[2]*100, wing_b))

        wing_txt.close()

if __name__ == '__main__':
    np.random.seed(1)

    '''
    RC* color hex
    get RGB for each rc
    '''
    # RC1 rgb mean [130 115  95]
    # hexes = '#75705D;#726449;#836E5B;#766A5A;#AB9379;#8F745F;#7D7161;#807863;#7F725F'
    # RC2 body rgb mean [226 222 223]   wing rgb mean [174 171 170]
    # hexes = '#D4D0D1;#C5C2C9;#E5DAD6;#E9E5E6;#F0E5E1;#DEDFE3;#FBFCF6;#E6E4E5;#DCDDE1' # body
    # hexes = '#ABA6A3;#BCBBB7;#A7A7A5;#B3B2AD;#B6B1AE;#B4AEAE;#B4B2B3;#A9A4AA;#9A9591' #wing
    # RC3 rgb mean [249 241 232]
    # hexes = '#faf0e3;#f7f1e3;#fdfaee;#feecdf;#feefe5;#ecece4;#fefefb;#f0e1e7;#fef7e9'
    # RC4 rgb mean [65 65 56]
    # hexes = '#49484D;#404447;#4D4845;#434544;#4F4F4F;#3F4134;#373528;#464236'
    # hexes = '#403e31;#423f30;#373220;#393b2d;#433d2d;#42484e;#464c4b;#4b4c4f'
    # RC5 rgb mean [182 126  56]
    # hexes = '#BF853B;#9F6832;#B07C30;#D28F40;#926023;#C98834;#B2895D;#B07527;#CB8E3E'
    # seeds = np.random.choice(range(5000), 450)
    # hex_list = [s for s in hexes.split(';')]
    # r_list = []
    # g_list = []
    # b_list = []
    # for hex in hex_list:
    #     (r, g, b) = ImageColor.getcolor(hex, "RGB")
    #     r_list.append(r)
    #     g_list.append(g)
    #     b_list.append(b)
    # r_mean = np.round(np.mean(r_list)).astype(np.int)
    # g_mean = np.round(np.mean(g_list)).astype(np.int)
    # b_mean = np.round(np.mean(b_list)).astype(np.int)
    # rgb_mean = np.array([r_mean, g_mean, b_mean])
    # print('rgb mean', rgb_mean)

    '''
    gaussian dist values for each rc
    '''
    ########## RC1
    # rare_cls = 1
    # body_mu = [130, 115, 95]
    # num_aft = int(450*7*3)
    # # ssig_list = [0, 15, 30, 45, 60]
    # # ssig_list = [0, 5, 10, 15, 20]
    # ssig_list = [0, 10**2, 20**2, 30**2, 40**2]
    # gaussian_disribution_of_color(body_mu, ssig_list, rare_cls, num_aft)
    #
    # ########## RC2
    # rare_cls = 2
    # body_mu = [226, 222, 223]
    # wing_mu = [174, 171, 170]
    # num_aft = int(450*7*3)
    # # ssig_list = [0, 15, 30, 45, 60]
    # # ssig_list = [0, 5, 10, 15, 20]
    # ssig_list = [0, 10**2, 20**2, 30**2, 40**2]
    # gaussian_disribution_of_color(body_mu, ssig_list, rare_cls, num_aft, wing_mu)
    #
    # ########## RC3
    # rare_cls = 3
    # body_mu =[249, 241, 232]
    # num_aft = int(450*7*3)
    # # ssig_list = [0, 15, 30, 45, 60]
    # # ssig_list = [0, 5, 10, 15, 20]
    # ssig_list = [0, 10**2, 20**2, 30**2, 40**2]
    # gaussian_disribution_of_color(body_mu, ssig_list, rare_cls, num_aft)
    #
    # ########## RC4
    # rare_cls = 4
    # body_mu =[65, 65, 56]
    # num_aft = int(450*7*3)
    # # ssig_list = [0, 15, 30, 45, 60]
    # # ssig_list = [0, 5, 10, 15, 20]
    # ssig_list = [0, 10**2, 20**2, 30**2, 40**2]
    # gaussian_disribution_of_color(body_mu, ssig_list, rare_cls, num_aft)
    #
    # ########## RC5
    # rare_cls = 5
    # body_mu =[182, 126, 56]
    # num_aft = int(450*7*3)
    # # ssig_list = [0, 15, 30, 45, 60]
    # # ssig_list = [0, 5, 10, 15, 20]
    # ssig_list = [0, 10**2, 20**2, 30**2, 40**2]
    # gaussian_disribution_of_color(body_mu, ssig_list, rare_cls, num_aft)


    # ########## CC1
    # ccid = 1
    # num_aft = int(450*7*3)
    # hex_file = '../../data_xview/1_cls/px23whr3_seed17/CC/CC_color/CC1_hex.csv'
    # df_hex = pd.read_csv(hex_file)
    # hex_list = df_hex['Body'].tolist()
    # rgb_mean, rgb_std = compute_rgb_by_hex(hex_list)
    # step_body = np.around(rgb_std/rgb_mean/4*0.9, decimals=2)
    # body_ssig_list = [i*step_body for i in range(5)]
    # print('step_body',step_body)
    # print('body_ssig_list', body_ssig_list)
    # gaussian_disribution_of_promu_rgb(rgb_mean, body_ssig_list, ccid, num_aft,  cmt='CC')

    # ########## CC2
    ccid = 2
    num_aft = int(450*7*3)
    hex_file = '../../data_xview/1_cls/px23whr3_seed17/CC/CC_color/CC2_hex.csv'
    df_hex = pd.read_csv(hex_file, sep='\t')
    body_hex_list = df_hex['Body'].tolist()
    rgb_mean, rgb_std = compute_rgb_by_hex(body_hex_list)
    step_body = np.around(rgb_std/rgb_mean/4*0.9, decimals=2)
    body_ssig_list = [i*step_body for i in range(5)]
    print('step_body',step_body)
    print('body_ssig_list', body_ssig_list)

    wing_hex_list = df_hex['Wing'].tolist()
    wing_rgb_mean, wing_rgb_std = compute_rgb_by_hex(wing_hex_list)
    step_wing = np.around(wing_rgb_std/wing_rgb_mean/4*0.9, decimals=2)
    wing_ssig_list = [i*step_wing for i in range(5)]
    print('step_wing',step_wing)
    print('wing_ssig_list', wing_ssig_list)
    # gaussian_disribution_of_promu_rgb(rgb_mean, body_ssig_list, ccid, num_aft, wing_mu=wing_rgb_mean, wing_ssig_list=wing_ssig_list, cmt='CC')
