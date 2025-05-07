import random
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from utils.Edge import Edge, sub_edge


class GraphDataset(Dataset):
    def __init__(
        self,
        opt,
        val,
        fake_rate,
        test=False,
        edge=None,
    ):
        self.opt = opt
        self.val = Edge(val)
        print(len(self.val.edge_set))
        self.test = test
        self.edges = edge
        self.fake_rate = fake_rate
        self.user_node_set = set(range(opt["number_user"]))
        self.item_node_set = set(range(opt["number_item"]))

    def __len__(self):
        return len(self.val.edge_list)

    def __getitem__(self, index):
        edge = self.val.edge_list[index]
        left, right, s = [edge[0]], [edge[1]], edge[2] == 1
        neighbour = np.array(
            self.edges.edge_abn[left[0]] + self.edges.edge_abp[left[0]]
        )
        if len(neighbour) == 0:
            left_n = []
        else:
            deg = self.edges.degb[neighbour]
            deg_idx = np.argsort(deg)[::-1]
            neighbour = neighbour[deg_idx].tolist()
            samples = neighbour[: self.opt["neighbours"]]
            left_n = list(set(samples) - set(right))
        neighbour = np.array(
            self.edges.edge_ban[right[0]] + self.edges.edge_bap[right[0]]
        )
        if len(neighbour) == 0:
            right_n = []
        else:
            deg = self.edges.dega[neighbour]
            deg_idx = np.argsort(deg)[::-1]
            neighbour = neighbour[deg_idx].tolist()
            samples = neighbour[: self.opt["neighbours"]]
            right_n = list(set(samples) - set(left))
        sub_0 = sub_edge(left, left_n, self.edges)
        sub_1 = sub_edge(right_n, left_n, self.edges)
        sub_2 = sub_edge(right_n, right, self.edges)
        edge_s = int(s)

        # 随机筛选假的节点和边就完了
        fake_left_number = int(len(left_n) * self.fake_rate)
        fake_user_set = self.item_node_set - set(right) - set(left_n)
        fake_left_n = random.sample(
            list(fake_user_set), min(fake_left_number, len(fake_user_set))
        )
        fake_right_number = int(len(right_n) * self.fake_rate)
        fake_item_set = self.user_node_set - set(left) - set(right_n)
        fake_right_n = random.sample(
            list(fake_item_set), min(fake_right_number, len(fake_item_set))
        )
        assert len(left_n) == len(fake_left_n)
        assert len(right_n) == len(fake_right_n)
        return {
            "left": left,
            "right": right,
            "left_n": left_n,
            "right_n": right_n,
            "sub_0": sub_0,
            "sub_1": sub_1,
            "sub_2": sub_2,
            "fake_left_n": fake_left_n,
            "fake_right_n": fake_right_n,
            "edge_s": edge_s,
        }


def pad_tensor(batch):
    max_size = [
        max(tensor.size(dim) for tensor in batch) for dim in range(batch[0].dim())
    ]
    batch_size = len(batch)
    background = torch.zeros([batch_size] + max_size, dtype=batch[0].dtype)
    for i, tensor in enumerate(batch):
        indices = tuple(slice(0, sz) for sz in tensor.size())
        background[i][indices] = tensor
    return background


def collate_fn(batch):
    batched_data = {key: [] for key in batch[0]}
    for item in batch:
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                batched_data[key].append(value)
            elif isinstance(value, list):
                batched_data[key].append(torch.tensor(value))
            else:
                batched_data[key].append(value)
    for key in batched_data:
        if isinstance(batched_data[key][0], torch.Tensor):
            if all(
                tensor.shape == batched_data[key][0].shape
                for tensor in batched_data[key]
            ):
                batched_data[key] = torch.stack(batched_data[key])
            else:
                batched_data[key] = pad_tensor(batched_data[key])
        else:
            batched_data[key] = torch.tensor(batched_data[key])

    return batched_data
