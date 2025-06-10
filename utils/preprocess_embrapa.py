# import os
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# import pickle
#
# def process_split_with_check(split_dir, verbose=True):
#     image_paths = []
#     labels = []
#
#     for label_name in sorted(os.listdir(split_dir)):
#         label_path = os.path.join(split_dir, label_name)
#         if not os.path.isdir(label_path):
#             continue
#
#         valid_files = [
#             fname for fname in os.listdir(label_path)
#             if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
#         ]
#
#         # Bỏ qua class rỗng
#         if len(valid_files) == 0:
#             if verbose:
#                 print(f"⚠️ Bỏ qua class trống: {label_name}")
#             continue
#
#         for fname in valid_files:
#             full_path = os.path.join(label_path, fname)
#             if os.path.exists(full_path):
#                 image_paths.append(os.path.join(label_name, fname))
#                 labels.append(label_name)
#             elif verbose:
#                 print(f"⚠️ Ảnh không tồn tại: {full_path}")
#
#     return pd.DataFrame({'image_path': image_paths, 'label': labels})
# # Đường dẫn
# base_dir = "./data/raw/embrapa"
#
# # Tạo dataframe cho từng tập
# train_df = process_split_with_check(os.path.join(base_dir, "train"))
# val_df = process_split_with_check(os.path.join(base_dir, "val"))
# test_df = process_split_with_check(os.path.join(base_dir, "test"))
#
# # Encode nhãn
# label_encoder = LabelEncoder()
# train_df['label_idx'] = label_encoder.fit_transform(train_df['label'])
# val_df['label_idx'] = label_encoder.transform(val_df['label'])
# test_df['label_idx'] = label_encoder.transform(test_df['label'])
#
# # Lưu lại
# output_dir = "./data/processed/embrapa"
# os.makedirs(output_dir, exist_ok=True)
# train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
# val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
# test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
#
# # Lưu label encoder
# with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
#     pickle.dump(label_encoder, f)
#
# print("✅ Đã tạo lại đầy đủ train/val/test CSV với kiểm tra ảnh.")

from torchvision import datasets, transforms

embrapa_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
