import argparse
import os
import time
from collections import OrderedDict, defaultdict

import jsonlines
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from modcloth import ModClothDataset
from model import SFNet
from utils import compute_metrics, load_config_from_json, to_var

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    sfnet, train_dataloader, valid_dataloader, num_epochs, save_model_path, model_config
):
    optimizer = torch.optim.Adam(
        sfnet.parameters(),
        lr=model_config["trainer"]["optimizer"]["lr"],
        weight_decay=model_config["trainer"]["optimizer"]["weight_decay"],
    )

    loss_criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    sfnet = sfnet.to(device)

    writer = SummaryWriter(os.path.join(save_model_path, "logs"))
    writer.add_text("model", str(sfnet))
    writer.add_text("args", str(args))

    dataloader_dict = {"train": train_dataloader, "valid": valid_dataloader}

    step = 0
    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    for epoch in range(num_epochs):
        for split in ["train", "valid"]:
            loss_tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == "train":
                sfnet.train()
            else:
                sfnet.eval()
                target_tracker = []
                pred_tracker = []

            for iteration, batch in enumerate(dataloader_dict[split]):

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                logits, pred_probs = sfnet(batch)

                # loss calculation
                loss = loss_criterion(logits, batch["fit"])

                # backward + optimization
                if split == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                # bookkeepeing
                loss_tracker["Total Loss"] = torch.cat(
                    (loss_tracker["Total Loss"], loss.view(1))
                )

                writer.add_scalar(
                    "%s/Total Loss" % split.upper(),
                    loss.item(),
                    epoch * len(dataloader_dict[split]) + iteration,
                )

                if iteration % model_config["logging"][
                    "print_every"
                ] == 0 or iteration + 1 == len(dataloader_dict[split]):
                    print(
                        "{} Batch Stats {}/{}, Loss={:.2f}".format(
                            split.upper(),
                            iteration,
                            len(dataloader_dict[split]) - 1,
                            loss.item(),
                        )
                    )

                if split == "valid":
                    target_tracker.append(batch["fit"].cpu().numpy())
                    pred_tracker.append(pred_probs.cpu().data.numpy())

            print(
                "%s Epoch %02d/%i, Mean Total Loss %9.4f"
                % (
                    split.upper(),
                    epoch + 1,
                    num_epochs,
                    torch.mean(loss_tracker["Total Loss"]),
                )
            )

            writer.add_scalar(
                "%s-Epoch/Total Loss" % split.upper(),
                torch.mean(loss_tracker["Total Loss"]),
                epoch,
            )

            # Save checkpoint
            if split == "train":
                checkpoint_path = os.path.join(
                    save_model_path, "E%i.pytorch" % (epoch + 1)
                )
                torch.save(sfnet.state_dict(), checkpoint_path)
                print("Model saved at %s" % checkpoint_path)

        if split == "valid":
            # not considering the last (incomplete) batch for metrics
            target_tracker = np.stack(target_tracker[:-1]).reshape(-1)
            pred_tracker = np.stack(pred_tracker[:-1], axis=0).reshape(
                -1, model_config["sfnet"]["num_targets"]
            )
            precision, recall, f1_score, accuracy, auc = compute_metrics(
                target_tracker, pred_tracker
            )

            writer.add_scalar("%s-Epoch/Precision" % split.upper(), precision, epoch)
            writer.add_scalar("%s-Epoch/Recall" % split.upper(), recall, epoch)
            writer.add_scalar("%s-Epoch/F1-Score" % split.upper(), f1_score, epoch)
            writer.add_scalar("%s-Epoch/Accuracy" % split.upper(), accuracy, epoch)
            writer.add_scalar("%s-Epoch/AUC" % split.upper(), auc, epoch)


def main(args):
    data_config = load_config_from_json(args.data_config_path)
    model_config = load_config_from_json(args.model_config_path)

    splits = ["train", "valid"]

    # データセット取得
    datasets = OrderedDict()
    for split in splits:
        datasets[split] = ModClothDataset(data_config, split=split)

    # dataloaderの構築
    train_dataloader = DataLoader(
        dataset=datasets["train"],
        batch_size=model_config["trainer"]["batch_size"],
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        dataset=datasets["valid"], batch_size=model_config["trainer"]["batch_size"],
    )

    # モデルの初期化
    sfnet = SFNet(model_config["sfnet"])

    # 保存先の設定
    ts = time.strftime("%Y-%b-%d-%H-%M-%S", time.gmtime())
    save_model_path = os.path.join(
        model_config["logging"]["save_model_path"],
        model_config["logging"]["run_name"] + ts,
    )
    os.makedirs(save_model_path)

    train(
        sfnet,
        train_dataloader,
        valid_dataloader,
        model_config["trainer"]["num_epochs"],
        save_model_path,
        model_config,
    )

    # モデル定義の保存
    with jsonlines.open(os.path.join(save_model_path, "config.jsonl"), "w") as fout:
        fout.write(model_config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config_path", type=str, default="configs/data.jsonnet")
    parser.add_argument(
        "--model_config_path", type=str, default="configs/model.jsonnet"
    )

    args = parser.parse_args()
    main(args)
