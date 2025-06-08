from matplotlib import transforms
from tensorflow.python.autograph.converters.call_trees import transform

transform = transforms.Compose(
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
)


class AppleDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_id"] + '.jpg')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(row['label_idx'])
        return image, label
