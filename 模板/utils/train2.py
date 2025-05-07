import torch

from utils import helper


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        schedule,
        scheduler_slow,
        loss_func,
        gpu_id: int,
        save_every: int,
        opt,
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
        nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.opt["max_grad_norm"]
        )
        loss.backward()
        print
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
        PATH = f"checkpoint_{epoch}.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def _init_helper(self, opt):
        model_id = opt["id"] if len(opt["id"]) > 1 else "0" + opt["id"]
        model_save_dir = opt["save_dir"] + "/" + model_id
        opt["model_save_dir"] = model_save_dir
        helper.ensure_dir(model_save_dir, verbose=True)
        # save config
        helper.save_config(opt, model_save_dir + "/config.json", verbose=True)
        file_logger = helper.FileLogger(
            model_save_dir + "/" + opt["log"],
            header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score",
        )
        # print model info
        helper.print_config(opt)
        format_str = "{}: step {}/{} (epoch {}/{}), loss = {:.6f} , lr: {:.6f}"
        file_logger.log(opt)
        return model_save_dir, file_logger, format_str

    def train(self, max_epochs: int):
        model_save_dir, file_logger, format_str = self._init_helper(self.opt)
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            self.scheduler.step()
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
