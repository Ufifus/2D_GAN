import numpy as np
import cv2
import os

from __init__ import databuilder_configs


path_npy_file = os.path.join(databuilder_configs['save_path'], 'dataset.npy')
path_preprocess_npy_file = os.path.join(databuilder_configs['save_path'], 'dataset.npz')
img_size = [int(dim) for dim in databuilder_configs['img_shape'].split(',')]
print(img_size)

with open(path_npy_file, 'rb') as npy_file:
    print(f'Open file {path_npy_file}...')
    dataset = np.load(npy_file, allow_pickle=True)

    preprocess_dataset = np.array([cv2.resize(img, img_size) for img in dataset[:, 0]])
    # preprocess_dataset = preprocess_dataset.astype(np.float32) - 127.5
    # preprocess_dataset = preprocess_dataset / 127.5
    preprocess_dataset = preprocess_dataset.astype(np.float32) / 255.0

    labels = dataset[:, 1]

    npy_file.close()

np.savez(path_preprocess_npy_file, X_train=preprocess_dataset, y_train=labels)
print('Preprocess image is Done, ready to train!')