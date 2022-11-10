from statistics import mean
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from __init__ import databuilder_configs


path_npy_file = os.path.join(databuilder_configs['save_path'], 'dataset.npy')
path_csv_file = os.path.join(databuilder_configs['save_path'], 'dataset.csv')

plt.figure(figsize=(9, 6))

with open(path_npy_file, 'rb') as npy_file:
    print(f'Open file {path_npy_file}...')
    dataset = np.load(path_npy_file, allow_pickle=True)

    widths = [img.shape[0] for img in dataset[:, 0]]
    heights = [img.shape[1] for img in dataset[:, 0]]

    mean_width = int(mean(widths))
    mean_height = int(mean(heights))

    print(f'mean width = {mean_width}; mean height = {mean_height}')

    n_classes = np.unique(dataset[:, 1])

    print(f'Consist {len(n_classes)} classes')


    n_classes_distibution = {k: 0 for k in n_classes}
    for label in dataset[:, 1]:
        n_classes_distibution[label] += 1
    counts, labels = zip(*sorted(n_classes_distibution.items()))
    plt.suptitle('n classes distribution')
    plt.bar(counts, labels)
    plt.show()

    npy_file.close()
