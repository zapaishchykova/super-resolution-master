import cv2
import numpy as np

path = r"E:\Anna\super-resolution-master\super-resolution-master\.div2k\images\DIV2K_valid_LR_bicubic"

lr = np.array(cv2.imread(path+'/0.png', cv2.IMREAD_ANYDEPTH))
lr_rgb = np.array(cv2.imread(path+'/rgb0.png',cv2.IMREAD_GRAYSCALE))

cv2.imwrite(path+'/1.png',lr[270:,370:870])
cv2.imwrite(path+'/rgb1.png',lr_rgb[270:,370:870])