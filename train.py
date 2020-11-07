import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from models.model import Net
from datasets.dataset import MnistDataset, collate_fn
from datasets.transforms import ToTensor


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prepare_data()
        self.setup_train()

    def prepare_data(self):
        train_val = MnistDataset(
            self.args.train_image_file,
            self.args.train_label_file,
            transform=transforms.Compose([ToTensor()]),
        )
        train_len = int(0.8 * len(train_val))
        train_ds, val_ds = torch.utils.data.random_split(
            train_val, [train_len, len(train_val) - train_len]
        )
        print("Train {}, val {}".format(len(train_ds), len(val_ds)))
        self.train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )

    def setup_train(self):
        self.model = Net().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        if not os.path.isdir(self.args.ckpt):
            os.mkdir(self.args.ckpt)

    def train_one_epoch(self):
        train_loss = 0.0
        self.model.train()
        for i, sample in enumerate(self.train_loader):
            X, Y_true = sample["X"].to(self.device), sample["Y"].to(self.device)
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, Y_true)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(self.train_loader)

    def evaluate(self):
        val_loss = 0.0
        self.model.eval()
        predicts = []
        truths = []
        with torch.no_grad():
            for i, sample in enumerate(self.val_loader):
                X, Y_true = sample["X"].to(self.device), sample["Y"].to(self.device)
                output = self.model(X)
                loss = self.criterion(output, Y_true)
                val_loss += loss.item()
                predicts.append(torch.argmax(output, dim=1))
                truths.append(Y_true)
        predicts = torch.cat(predicts, dim=0)
        truths = torch.cat(truths, dim=0)
        acc = torch.sum(torch.eq(predicts, truths))
        return acc / len(predicts), val_loss / (len(self.val_loader))

    def run(self):
        min_loss = 10e4
        max_acc = 0
        for epoch in range(self.args.epochs):
            train_loss = self.train_one_epoch()
            val_acc, val_loss = self.evaluate()

            if val_acc > max_acc:
                max_acc = val_acc
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.args.ckpt,
                        "{}_{}_{:.4f}.pth".format(self.args.name, epoch, max_acc),
                    ),
                )
            print(
                "Epoch {}, loss {:.4f}, val_acc {:.4f}".format(
                    epoch, train_loss, val_acc
                )
            )


def main(args):
    trainer = Trainer(args)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch Mnist training script")
    parser.add_argument("--name", default=None, type=str, help="Name script")
    parser.add_argument("--ckpt", default=None, type=str, help="Checkpoint directory")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--epochs", default=80, type=int, help="Number of epochs to train"
    )
    parser.add_argument(
        "--train_image_file", default=None, type=str, help="Train image file"
    )
    parser.add_argument(
        "--train_label_file", default=None, type=str, help="Train label file"
    )
    args = parser.parse_args()
    print(args)
    main(args)
