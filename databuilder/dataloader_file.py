import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from __init__ import databuilder_configs


"""Test module for testing saving files with datasets"""

def plot_imgs(imgs, titles=[]):
    ## multiple images
    fig, ax = plt.subplots(nrows=1, ncols=len(imgs), sharex=False, sharey=False, figsize=(4 * len(imgs), 10))
    if len(titles) == 1:
        fig.suptitle(titles[0], fontsize=15)
    for i, img in enumerate(imgs):
        print(img.shape)
        ax[i].imshow(img)
        if len(titles) > 1:
            ax[i].set(title=titles[i])
    plt.show()


with open(os.path.join(databuilder_configs['save_path'], 'dataset.npz'), 'rb') as f:
    tensor = np.load(f, allow_pickle=True)
    train = tensor['X_train']
    print(train.shape)
    plot_imgs(train[:5], ['fdf'])

    f.close()