import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from DDP.single_gpu import load_train_objs
from datautils import MyTrainDataset

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int
    ):
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.gpu_id = gpu_id
        self.save_every = save_every
        pass

    def _run_batch(self, source, targets):
        "前向，损失，反传"
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        
    def _run_epoch(self, epoch):
        "加载数据，循环，batch"
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
            
    def _save_checkpoint(self, epoch):
        "加载权重保存"
        ckp = self.model.state_dict()
        PATH= f'checkpoint_{epoch}.pt'
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
        
    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def load_train_objs():
    dataset = MyTrainDataset(2048)
    model = torch.nn.Linear(20, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def main(device, total_epochs, save_every, batch_size):
    # 组件准备
    dataset, model, optimizer = load_train_objs()
    train_data = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    trainer = Trainer(model, train_data, optimizer, device, save_every)
    trainer.train(total_epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "total_epochs", type=int, help="Total epochs to train the model"
    )
    parser.add_argument("save_every", type=int, help="How often to save a snapshot")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Input batch size on each device (default: 32)",
    )
    args = parser.parse_args()

    device = 0  # shorthand for cuda:0
    main(device, args.total_epochs, args.save_every, args.batch_size)
