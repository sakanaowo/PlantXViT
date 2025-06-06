import os

from utils.config_loader import load_config

config = load_config()
data_dir = config['dataset']['data_dir']
path = os.path.join(data_dir, 'train')
class_names = sorted(os.listdir(path))
print(class_names)
