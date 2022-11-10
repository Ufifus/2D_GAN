import os
import configparser


"""Configs.py contains environment variables for work currently GAN-app"""

configs = configparser.ConfigParser()
configs.read('configs_venv.ini')

# DATABUILDER configs
databuilder_configs = configs['datapreprocessing']
print([[k, v] for k, v in databuilder_configs.items()])

data_dir = os.path.abspath('data')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# Dataloader
# load path with images
load_path = databuilder_configs['load_path']
if load_path == '':
    load_path = os.path.abspath('data/images')
    if not os.path.exists(load_path):
        os.mkdir(load_path)
    databuilder_configs['load_path'] = load_path
elif not os.path.exists(load_path):
    raise Exception(f'load path = {load_path} doesnt exist')

# save path with images, use for save new generate images, results epochs and npy files
save_path = databuilder_configs['save_path']
if save_path == '':
    save_path = os.path.abspath('data/save_data')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    databuilder_configs['save_path'] = save_path
elif not os.path.exists(save_path):
    raise Exception(f'saved path = {save_path} doesnt exist')

#able formats images
able_formats = [able_format for able_format in databuilder_configs['ableformats'].split(',')]
if len(able_formats) == 0:
    raise Exception('Please append formats images in configs')

#resize size
if databuilder_configs['img_shape'] == '':
    databuilder_configs['img_shape'] = '300,300'
else:
    try:
        img_size = (int(dim) for dim in databuilder_configs['img_shape'].split(','))
    except:
        raise Exception('Size image in configs is not int type')


#MODELBUILDER configs
modelbuilder_configs = configs['modelconfig']
print([[k, v] for k, v in modelbuilder_configs.items()])


