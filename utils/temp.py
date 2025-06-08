# Giả sử log_text chứa toàn bộ đoạn log bạn dán bên trên (dạng string)
from utils.helper import parse_log_file

train_loss, train_acc, val_loss, val_acc = parse_log_file('../outputs/logs/train_attempt1.txt')

# Vẽ biểu đồ
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Accuracy
# plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Loss
# plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
