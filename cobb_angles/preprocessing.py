import numpy as np
from PIL import Image
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import ndimage, misc


def open_XR(root_dir, file_name):
    """Given the path and file name, this function opens an XR image and returns a numpy array of shape [height, width, 1]"""
    image = Image.open(os.path.join(root_dir, file_name))
    image = image.convert('RGB')
    return image


def zeropad(img, height=4000, width=1500):
    pad_image = np.zeros((height, width, 3))
    beg_height = (height - img.shape[0]) // 2
    end_height = beg_height + img.shape[0]
    beg_width = (width - img.shape[1]) // 2
    end_width = beg_width + img.shape[1]
    pad_image[beg_height:end_height, beg_width:end_width, :] = img
    return pad_image


def preprocess(root_dir, file_name):
    image = open_xr(root_dir, file_name)
    return image
