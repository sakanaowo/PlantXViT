from torch.utils.data import DataLoader

batch_size = config["training"]["batch_size"]

train_dataset = AppleDataset(train_df, img_dir, transform)
val_dataset = AppleDataset(val_df, img_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
