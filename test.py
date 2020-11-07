import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from models.model import Net
from datasets.dataset import MnistDataset, collate_fn
from datasets.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    test_ds = MnistDataset(
        args.test_image_file,
        args.test_label_file,
        transform=transforms.Compose([ToTensor()]),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )
    model = Net().to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    predicts = []
    truths = []
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            X, Y_true = sample["X"].to(device), sample["Y"].to(device)
            output = model(X)
            predicts.append(torch.argmax(output, dim=1))
            truths.append(Y_true)
    predicts = torch.cat(predicts, dim=0)
    truths = torch.cat(truths, dim=0)
    acc = torch.sum(torch.eq(predicts, truths))
    print("Acc: {:.4f}".format(acc / len(predicts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch Mnist testing script")
    parser.add_argument("--name", default=None, type=str, help="Name script")
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint file")
    parser.add_argument(
        "--test_image_file", default=None, type=str, help="Path to input file"
    )
    parser.add_argument(
        "--test_label_file", default=None, type=str, help="Path to input file"
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")

    args = parser.parse_args()
    main(args)
