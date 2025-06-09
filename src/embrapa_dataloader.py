from torchvision import transforms
from torch.utils.data import DataLoader

# Transform dùng chung cho cả 3 tập
embrapa_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Root thư mục chứa ảnh
image_root = "./data/raw/embrapa"

# Tạo dataset
train_dataset = EmbrapaDataset("./data/processed/embrapa_train.csv", image_root, transform=embrapa_transform)
val_dataset = EmbrapaDataset("./data/processed/embrapa_val.csv", image_root, transform=embrapa_transform)
test_dataset = EmbrapaDataset("./data/processed/embrapa_test.csv", image_root, transform=embrapa_transform)

# Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)
