import re

def parse_log_file(filepath):
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    pattern = r"Train Loss: ([0-9.]+), Acc: ([0-9.]+) \| Val Loss: ([0-9.]+), Acc: ([0-9.]+)"

    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                train_loss = float(match.group(1))
                train_acc = float(match.group(2))
                val_loss = float(match.group(3))
                val_acc = float(match.group(4))

                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc)

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list
