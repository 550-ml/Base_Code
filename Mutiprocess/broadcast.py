import os
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rankid, size, func, backend="gloo") -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "65534"

    dist.init_process_group(backend=backend, rank=rankid, world_size=size)

    func(rankid, size)


# def run(rank_id, size):
#     tensor = torch.arange(2) + rank_id + 1
#     print(f"before broadcast Rank {rank_id} has data {tensor}")

#     # 广播
#     dist.broadcast(tensor=tensor, src=0)
#     print(f"after broadcast Rank {rank_id} has data {tensor}")
def run(rank_id, size):
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank_id
    print('before scatter',' Rank ', rank_id, ' has data ', tensor)
    if rank_id == 0:
        scatter_list = [torch.tensor([0,0]), torch.tensor([1,1]), torch.tensor([2,2]), torch.tensor([3,3])]
        print('scater list:', scatter_list)
        dist.scatter(tensor, src = 0, scatter_list=scatter_list)
    else:
        dist.scatter(tensor, src = 0)
    print('after scatter',' Rank ', rank_id, ' has data ', tensor)

if __name__ == "__main__":
    size = 4
    process_list: List[mp.Process] = []

    for rank in range(size):
        # 每一个线程都要初始化
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()
