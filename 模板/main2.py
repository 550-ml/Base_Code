from arrow import get
import torch
from configs.config import seed_everything
from torch.utils.data import DataLoader
from utils import helper

from DDP.single_gpu import load_train_objs


def load_train_objs(args):
    dataset = None
    model = None
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scheduler_slow = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    loss_func = torch.nn.BCELoss()
    return dataset, model, optimizer, scheduler, scheduler_slow, loss_func


def main(device, total_epochs, save_every, batch_size, opt):
    # 组件准备
    dataset, model, optimizer, scheduler, scheduler_slow, loss_func = load_train_objs()
    train_data = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    file_logger = 
    trainer = Trainer(model, train_data, optimizer, device, save_every)
    trainer.train(total_epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple distributed training job")
    # dataset part
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, default="amazon-book-1")

    # model part
    parser.add_argument(
        "--neighbours", type=int, default=200, help="Number of neighbours."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size."
    )
    parser.add_argument("--struct_rate", type=float, default=0.000001)
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=128,
        help="Initialize network embedding dimension.",
    )
    parser.add_argument("--GNN", type=int, default=2, help="The layer of encoder.")
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="GNN layer dropout rate."
    )
    parser.add_argument("--leakey", type=float, default=0.1)

    # log part
    parser.add_argument(
        "--load",
        dest="load",
        action="store_true",
        default=False,
        help="Load pretrained model.",
    )
    parser.add_argument(
        "--log_step", type=int, default=200, help="Print log every k steps."
    )
    parser.add_argument(
        "--log", type=str, default="logs.txt", help="Write training log to file."
    )
    parser.add_argument(
        "--save_epoch",
        type=int,
        default=100,
        help="Save model checkpoints every k epochs.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved_models",
        help="Root dir for saving models.",
    )
    parser.add_argument(
        "--id", type=str, default="01", help="Model ID under which to save models."
    )

    # training part
    parser.add_argument(
        "--epoch", type=int, default=100, help="Number of total training epochs."
    )
    parser.add_argument("--cpu", action="store_true", help="Ignore CUDA.")
    parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
    parser.add_argument(
        "--optim",
        choices=["sgd", "adagrad", "adam", "adamax"],
        default="adam",
        help="Optimizer: sgd, adagrad, adam or adamax.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Applies to sgd and adagrad."
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.9, help="Learning rate decay rate."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (L2 loss on parameters).",
    )
    parser.add_argument(
        "--decay_epoch",
        type=int,
        default=5,
        help="Decay learning rate after this epoch.",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=5.0, help="Gradient clipping."
    )
    args = parser.parse_args()

    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)
    opt = vars(args)
    seed_everything(opt["seed"])
    device = torch.device("cuda" if args.cuda else "cpu")
    main(device, args.total_epochs, args.save_every, args.batch_size, opt = opt)
