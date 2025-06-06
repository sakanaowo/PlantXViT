import os
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import load_img, img_to_array
import pickle
from utils.config_loader import load_config

# Load config
config = load_config()
img_size = tuple(config["dataset"]["image_size"])
data_dir = config["dataset"]["data_dir"]
save_dir = config["output"]["save_dir"]
label_path = config["dataset"]["label_encoder"]

os.makedirs(save_dir, exist_ok=True)

def load_images_and_labels(split_dir):
    images, labels = [], []
    class_names = sorted(os.listdir(split_dir))

    for class_name in class_names:
        class_path = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img = load_img(os.path.join(class_path, fname), target_size=img_size)
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(class_name)

    return np.array(images), np.array(labels)

def process_split(split):
    path = os.path.join(data_dir, split)
    X, y = load_images_and_labels(path)
    return X, y

# Process all splits
trainX, trainY = process_split("train")
valX, valY = process_split("val")
testX, testY = process_split("test")

# Encode labels
encoder = LabelEncoder()
trainY_enc = encoder.fit_transform(trainY)
valY_enc = encoder.transform(valY)
testY_enc = encoder.transform(testY)

trainY_oh = to_categorical(trainY_enc)
valY_oh = to_categorical(valY_enc)
testY_oh = to_categorical(testY_enc)

# Save processed data
np.savez_compressed(os.path.join(save_dir, "train.npz"), X=trainX, y=trainY_oh)
np.savez_compressed(os.path.join(save_dir, "val.npz"), X=valX, y=valY_oh)
np.savez_compressed(os.path.join(save_dir, "test.npz"), X=testX, y=testY_oh)

# Save encoder
with open(label_path, "wb") as f:
    pickle.dump(encoder, f)

print("✅ Dữ liệu đã được tiền xử lý và lưu vào:", save_dir)
