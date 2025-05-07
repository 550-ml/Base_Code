from sympy import im
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Attntion(nn.Module):
    def __init__(self, input_dim, head=4):
        super(Attntion, self).__init__()
        self.bt_pre = nn.Linear(input_dim, 6)
        self.bt_cur = nn.Linear(input_dim, 6)
        self.fcc = nn.Sequential(
            GELU(),
            nn.Linear(8, head),
        )
        self.fcg = nn.Sequential(
            GELU(),
            nn.Linear(8, input_dim),
        )
        self.head = head
        self.dim = input_dim
        self.fuse = nn.Sequential(
            GELU(),
            nn.Linear(12, 4),
        )
        self.ffn = nn.Sequential(
            GELU(),
            nn.Linear(12, 8),
        )

    def forward(self, prev, curr, edges):
        # edges: (b, n_pre, n_curr)
        bt_pre = self.bt_pre(prev)  # (b, n, 6)
        bt_cur = self.bt_cur(curr)  # (b, n, 6)
        shape = (
            bt_pre.shape[0],
            bt_pre.shape[1],
            bt_cur.shape[1],
            bt_pre.shape[2],
        )  # (b, n, n, 6)
        bt_pre = bt_pre.unsqueeze(2).expand(shape)  # (b, n, n, 6)
        bt_cur = bt_cur.unsqueeze(1).expand(shape)  # (b, n, n, 6)
        c = torch.cat((bt_pre, bt_cur), dim=-1)  # (b, n, n, 12)
        c = self.fuse(c)  # (b, n, n, 4)
        c[edges == 0] = 0  # 在n,n维度置为0
        edges1, edges2 = torch.sum(edges, dim=1) + 1, torch.sum(edges, dim=2) + 1
        edges1[edges1 == 0] = 1
        edges2[edges2 == 0] = 1
        d = torch.sum(c, dim=1).unsqueeze(1).expand(c.shape) / edges1.unsqueeze(
            1
        ).unsqueeze(-1)
        e = torch.max(c, dim=1)[0].unsqueeze(1).expand(c.shape)
        fused = self.ffn(
            torch.cat((c, d, e), dim=-1)
        )  # (batch_size, num_nodes, num_nodes, 8)
        c = self.fcc(fused).transpose(
            -2, -3
        )  # (batch_size, num_nodes, num_nodes, head)
        mask = edges.transpose(-1, -2)
        res_list = []
        for i in range(self.head):
            c_true = c[:, :, :, i].clone()
            if c_true.shape[2] != 1:
                c_true[mask == 0] = -9e15
                c_true = F.softmax(
                    c_true, dim=2
                ).clone()  # (batch_size, num_nodes, num_nodes)
            c_true[mask == 0] = 0
            res = torch.bmm(
                c_true,
                prev[:, :, self.dim // self.head * i : self.dim // self.head * (i + 1)],
            )
            res_list.append(res)
        res = torch.cat(res_list, dim=-1)
        fused = fused.clone()
        fused[edges == 0] = 0
        res = res + self.fcg(torch.sum(fused, dim=1) / edges1.unsqueeze(-1))
        return res


class GELU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )
