import imageio
import numpy as np
from matplotlib import pyplot as plt

# file1 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_59_RGB.jpg'
# file2 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_60_RGB.jpg'
file1 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_75_RGB.jpg'
file2 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_76_RGB.jpg'

img_file1 = imageio.imread(file1)
img_file2 = imageio.imread(file2)

res = 0.3
size = img_file1.shape[0]
step = size//2

common1 = img_file1[:, step:]
common2 = img_file2[:, :step]

plt.figure(figsize=(15, 8))
plt.subplot(131)
plt.imshow(common1)

plt.subplot(132)
plt.imshow(common2)

plt.subplot(133)
plt.imshow(common2-common1)

plt.tight_layout()
plt.show()

# plt.savefig('/media/lab/Yang/data/synthetic_data/Airplanes/59_60common_change.jpg')

plt.savefig('/media/lab/Yang/data/synthetic_data/Airplanes/75_76common_change.jpg')
