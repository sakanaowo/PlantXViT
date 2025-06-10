import os.path

from torchvision import transforms
from torch.utils.data import DataLoader

from utils.preprocess_embrapa import embrapa_transform

# Root thư mục chứa ảnh
image_root = "./data/raw/embrapa"

# Tạo dataset
train_dataset = datasets.ImageFolder(os.path.join(image_root, 'train'), transform=embrapa_transform)
val_dataset = datasets.ImageFolder(os.path.join(image_root, 'val'), transform=embrapa_transform)
test_dataset = datasets.ImageFolder(os.path.join(image_root, 'test'), transform=embrapa_transform)

# Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
