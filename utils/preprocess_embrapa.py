import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle


def preprocess_embrapa(split_dir):
    image_ids = []
    labels = []
    for label_name in sorted(os.listdir(split_dir)):
        label_path = os.path.join(split_dir, label_name)
        if os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                image_ids.append(os.path.join(label_path, fname))
                labels.append(label_name.lower())

    return pd.DataFrame({'image_path': image_ids, 'label': labels})


data_root = config['dataset']['embrapa']['data_dir']

# dataframe for each split
df_train = preprocess_embrapa(os.path.join(data_root, 'train'))
df_val = preprocess_embrapa(os.path.join(data_root, 'val'))
df_test = preprocess_embrapa(os.path.join(data_root, 'test'))

# encode label

label_encoder = LabelEncoder()
df_train['label_idx'] = label_encoder.fit_transform(df_train['label'])
df_val['label_idx'] = label_encoder.transform(df_val['label'])
df_test['label_idx'] = label_encoder.transform(df_test['label'])

# save encoder
os.makedirs('./data/processed/embrapa', exist_ok=True)
with open('./data/processed/embrapa/train.pkl', 'wb') as f:
    pickle.dump(df_train, f)

# save csv
df_train.to_csv('./data/processed/embrapa/train.csv', index=False)
df_val.to_csv('./data/processed/embrapa/val.csv', index=False)
df_test.to_csv('./data/processed/embrapa/test.csv', index=False)
