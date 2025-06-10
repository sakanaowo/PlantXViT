# # Giả sử log_text chứa toàn bộ đoạn log bạn dán bên trên (dạng string)
# from utils.helper import parse_log_file
#
# train_loss, train_acc, val_loss, val_acc = parse_log_file('../outputs/apple/logs/train_attempt1.txt')
#
# # Vẽ biểu đồ
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(12, 5))
#
# # Accuracy
# # plt.subplot(1, 2, 1)
# plt.plot(train_acc, label='Train Acc')
# plt.plot(val_acc, label='Val Acc')
# plt.title('Accuracy per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.show()
#
# # Loss
# # plt.subplot(1, 2, 2)
# plt.plot(train_loss, label='Train Loss')
# plt.plot(val_loss, label='Val Loss')
# plt.title('Loss per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.tight_layout()
# plt.show()

import re
log_text="""
Epoch 1/50

Training: 100%|██████████| 1851/1851 [01:15<00:00, 24.49it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.84it/s]

Train Loss: 3.3801 | Acc: 0.3026
Val   Loss: 2.7352 | Acc: 0.4011
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 2/50

Training: 100%|██████████| 1851/1851 [01:17<00:00, 23.93it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 49.20it/s]

Train Loss: 2.3547 | Acc: 0.4647
Val   Loss: 2.0359 | Acc: 0.5218
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 3/50

Training: 100%|██████████| 1851/1851 [01:18<00:00, 23.59it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 49.17it/s]

Train Loss: 1.8078 | Acc: 0.5763
Val   Loss: 1.6736 | Acc: 0.6189
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 4/50

Training: 100%|██████████| 1851/1851 [01:17<00:00, 23.99it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.52it/s]

Train Loss: 1.4754 | Acc: 0.6446
Val   Loss: 1.3858 | Acc: 0.6567
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 5/50

Training: 100%|██████████| 1851/1851 [01:15<00:00, 24.38it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.57it/s]

Train Loss: 1.2590 | Acc: 0.6840
Val   Loss: 1.2172 | Acc: 0.6944
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 6/50

Training: 100%|██████████| 1851/1851 [01:16<00:00, 24.21it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.84it/s]

Train Loss: 1.1021 | Acc: 0.7181
Val   Loss: 1.0689 | Acc: 0.7154
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 7/50

Training: 100%|██████████| 1851/1851 [01:16<00:00, 24.28it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 49.79it/s]

Train Loss: 0.9837 | Acc: 0.7430
Val   Loss: 0.9726 | Acc: 0.7323
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 8/50

Training: 100%|██████████| 1851/1851 [01:15<00:00, 24.42it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.73it/s]

Train Loss: 0.8982 | Acc: 0.7629
Val   Loss: 0.9167 | Acc: 0.7486
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 9/50

Training: 100%|██████████| 1851/1851 [01:17<00:00, 23.78it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 49.00it/s]

Train Loss: 0.8217 | Acc: 0.7797
Val   Loss: 0.8974 | Acc: 0.7493
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 10/50

Training: 100%|██████████| 1851/1851 [01:16<00:00, 24.28it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.78it/s]

Train Loss: 0.7647 | Acc: 0.7932
Val   Loss: 0.7873 | Acc: 0.7827
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 11/50

Training: 100%|██████████| 1851/1851 [01:15<00:00, 24.64it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.71it/s]

Train Loss: 0.7029 | Acc: 0.8075
Val   Loss: 0.7190 | Acc: 0.8034
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 12/50

Training: 100%|██████████| 1851/1851 [01:16<00:00, 24.36it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.29it/s]

Train Loss: 0.6636 | Acc: 0.8162
Val   Loss: 0.7502 | Acc: 0.7901

Epoch 13/50

Training: 100%|██████████| 1851/1851 [01:15<00:00, 24.41it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.16it/s]

Train Loss: 0.6201 | Acc: 0.8264
Val   Loss: 0.7097 | Acc: 0.7995

Epoch 14/50

Training: 100%|██████████| 1851/1851 [01:17<00:00, 23.86it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 49.47it/s]

Train Loss: 0.5852 | Acc: 0.8364
Val   Loss: 0.6622 | Acc: 0.8064
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 15/50

Training: 100%|██████████| 1851/1851 [01:17<00:00, 23.87it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 49.04it/s]

Train Loss: 0.5574 | Acc: 0.8414
Val   Loss: 0.6192 | Acc: 0.8229
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 16/50

Training: 100%|██████████| 1851/1851 [01:17<00:00, 23.96it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.89it/s]

Train Loss: 0.5315 | Acc: 0.8470
Val   Loss: 0.6362 | Acc: 0.8170

Epoch 17/50

Training: 100%|██████████| 1851/1851 [01:18<00:00, 23.48it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 47.78it/s]

Train Loss: 0.5067 | Acc: 0.8536
Val   Loss: 0.5998 | Acc: 0.8266
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 18/50

Training: 100%|██████████| 1851/1851 [01:16<00:00, 24.05it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.14it/s]

Train Loss: 0.4833 | Acc: 0.8605
Val   Loss: 0.6043 | Acc: 0.8231

Epoch 19/50

Training: 100%|██████████| 1851/1851 [01:18<00:00, 23.59it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.20it/s]

Train Loss: 0.4694 | Acc: 0.8654
Val   Loss: 0.5709 | Acc: 0.8341
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 20/50

Training: 100%|██████████| 1851/1851 [01:15<00:00, 24.49it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.12it/s]

Train Loss: 0.4461 | Acc: 0.8701
Val   Loss: 0.5699 | Acc: 0.8346
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 21/50

Training: 100%|██████████| 1851/1851 [01:16<00:00, 24.05it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 49.72it/s]

Train Loss: 0.4284 | Acc: 0.8751
Val   Loss: 0.6039 | Acc: 0.8237

Epoch 22/50

Training: 100%|██████████| 1851/1851 [01:17<00:00, 23.88it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.53it/s]

Train Loss: 0.4131 | Acc: 0.8790
Val   Loss: 0.5662 | Acc: 0.8341

Epoch 23/50

Training: 100%|██████████| 1851/1851 [01:17<00:00, 23.85it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.42it/s]

Train Loss: 0.4023 | Acc: 0.8807
Val   Loss: 0.5573 | Acc: 0.8283

Epoch 24/50

Training: 100%|██████████| 1851/1851 [01:16<00:00, 24.17it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 49.38it/s]

Train Loss: 0.3915 | Acc: 0.8835
Val   Loss: 0.5290 | Acc: 0.8425
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 25/50

Training: 100%|██████████| 1851/1851 [01:16<00:00, 24.35it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 49.24it/s]

Train Loss: 0.3735 | Acc: 0.8900
Val   Loss: 0.5146 | Acc: 0.8510
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 26/50

Training: 100%|██████████| 1851/1851 [01:17<00:00, 23.94it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 49.49it/s]

Train Loss: 0.3659 | Acc: 0.8924
Val   Loss: 0.5108 | Acc: 0.8506

Epoch 27/50

Training: 100%|██████████| 1851/1851 [01:16<00:00, 24.20it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.27it/s]

Train Loss: 0.3527 | Acc: 0.8951
Val   Loss: 0.5477 | Acc: 0.8366

Epoch 28/50

Training: 100%|██████████| 1851/1851 [01:17<00:00, 23.86it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 49.02it/s]

Train Loss: 0.3394 | Acc: 0.8998
Val   Loss: 0.5301 | Acc: 0.8449

Epoch 29/50

Training: 100%|██████████| 1851/1851 [01:17<00:00, 23.83it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.95it/s]

Train Loss: 0.3372 | Acc: 0.8993
Val   Loss: 0.5292 | Acc: 0.8449

Epoch 30/50

Training: 100%|██████████| 1851/1851 [01:17<00:00, 23.84it/s]
Evaluating: 100%|██████████| 466/466 [00:09<00:00, 48.56it/s]

Train Loss: 0.3245 | Acc: 0.9036
Val   Loss: 0.5124 | Acc: 0.8487
Early stopping at epoch 30
"""

# Tìm tất cả các dòng Train và Val
train_lines = re.findall(r'Train Loss: ([\d\.]+) \| Acc: ([\d\.]+)', log_text)
val_lines = re.findall(r'Val\s+Loss: ([\d\.]+) \| Acc: ([\d\.]+)', log_text)

# Tách ra các list số
train_losses = [float(l[0]) for l in train_lines]
train_accuracies = [float(l[1]) for l in train_lines]
val_losses = [float(l[0]) for l in val_lines]
val_accuracies = [float(l[1]) for l in val_lines]

import matplotlib.pyplot as plt

epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(14, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Val Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
