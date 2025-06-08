import panda as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from torchvision import transform
import pickle

from utils.config_loader import load_config

# load config
config = load_config()
apple_config = config['dataset']['apple']
img_dir = apple_config['data_dir']
csv_path = apple_config['csv_path']
label_encoder_path = apple_config['label_encoder']
image_size = tuple(apple_config['image_size'])

# read csv, preprocess label
df = pd.read_csv(csv_path)
df['label'] = df[['healthy', 'multiple_diseases', 'rust', 'scab']].idxmax(axis=1)


# encode label
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# save label encoded
os.makedirs(os.path.dirname(label_encoder_path), exist_ok=True)
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)

# split train/val
train_df, val_df = train_test_split(df, test_size=0.2,
                                    stratify=df['label'],
                                    random_state=42)
