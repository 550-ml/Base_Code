import os
import numpy as np


def load_data(data_dir, dataset_name):
    train_file_path = os.path.join(data_dir, f"{dataset_name}_training.txt")
    val_file_path = os.path.join(data_dir, f"{dataset_name}_validation.txt")
    test_file_path = os.path.join(data_dir, f"{dataset_name}_testing.txt")
    # if not os.path.exists(val_file_path):
    #     os.rename(os.path.join(data_prefix, f'{dataset_name}_val.txt'),val_file_path)
    # if not os.path.exists(test_file_path):
    #     os.rename(os.path.join(data_prefix, f'{dataset_name}_test.txt'),test_file_path)

    train_edgelist = []
    with open(train_file_path) as f:
        for ind, line in enumerate(f):
            if ind == 0:
                continue
            a, b, s = map(int, line.split("\t"))
            train_edgelist.append((a, b, s))

    val_edgelist = []
    with open(val_file_path) as f:
        for ind, line in enumerate(f):
            if ind == 0:
                continue
            a, b, s = map(int, line.split("\t"))
            val_edgelist.append((a, b, s))

    test_edgelist = []
    with open(test_file_path) as f:
        for ind, line in enumerate(f):
            if ind == 0:
                continue
            a, b, s = map(int, line.split("\t"))
            test_edgelist.append((a, b, s))

    return np.array(train_edgelist), np.array(val_edgelist), np.array(test_edgelist)
