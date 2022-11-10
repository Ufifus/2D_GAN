import pandas as pd
import numpy as np
import os
import cv2
from __init__ import databuilder_configs


able_formats = [able_format for able_format in databuilder_configs['ableformats'].split(',')]


def check_load_path(load_path):
    """Check folder contain any folders
    Format to load path load_dir/(class_1, class_2, ...)"""
    folders = list(filter(lambda x: not os.path.isdir(x), os.listdir(load_path)))
    if len(folders) == 0:
        raise Exception(f'load dir {load_path} is empty')
    return folders


def read_folder(labels, folders, load_path):
    """Read every class dir and return generator"""
    for label, folder_name in zip(labels, folders):
        folder_path = os.path.join(load_path, folder_name)
        print('Current folder = ', folder_path)
        for img_name in os.listdir(folder_path):
            img_format = img_name.split('.')[-1]
            if img_format in able_formats:
                yield folder_path, img_name, label


def read_img(img_params):
    for folder, img_name in img_params:
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        try:
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                yield img
        except Exception as e:
            print(f'failed on {img_path}, error {e}')
            yield np.nan


if __name__ == '__main__':
    print('Starting data load...')

    # Path to folder which contains images
    print('Checking load dir')
    load_path = databuilder_configs['load_path']

    folders = check_load_path(load_path)
    labels = [folder[-1] for folder in folders]
    print('labels = ', labels)
    print('folders = ', folders)

    print('Generate pandas dataframe witn images')
    dtf = pd.DataFrame(read_folder(labels, folders, load_path))
    dtf.columns = ['folder', 'img_name', 'label']
    dtf['img'] = np.array(read_img(dtf[['folder', 'img_name']].apply(tuple, axis=1)))
    dtf["y"] = np.array(dtf["label"].factorize(sort=True)[0], dtype=int)

    print(dtf['img'].shape, dtf['y'].shape)

    dtf = dtf[['folder', 'img_name', 'img', 'label', 'y']]

    print('Clear void imgs')
    dtf.dropna(subset=["img"], inplace=True)

    # print('Result == \n', dtf)

    print('Saving dtf as csv file and npy file for tensors imgs')
    dtf[['folder', 'img_name', 'label']].to_csv(os.path.join(databuilder_configs['save_path'], 'dataset.csv'), index=False)
    np.save(os.path.join(databuilder_configs['save_path'], 'dataset.npy'), np.array(
        np.array([[img, y] for img, y in zip(dtf['img'], dtf['y'])])
    ))

    print('Done.')



