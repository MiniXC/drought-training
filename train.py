import click
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, mean_absolute_error
import shap
import plotly.figure_factory as ff
import wandb

from utils import interpolate_nans, class2id, id2class, date_encode, loadXY, normalize
from lstm import DroughtNetLSTM
from transformer import DroughtNetTransformer

@click.command()
@click.option("--model-type", default='lstm')
@click.option("--use-past", default=True)
@click.option("--use-previous-year", default=False)
@click.option("--use-season", default=True)
@click.option("--use-static", default=False)
@click.option("--window-size", default=180)
@click.option("--batch-size", default=128)
@click.option("--shuffle", default=True)
@click.option("--lr", default=7e-5)
@click.option("--one-cycle", default=True)
@click.option("--output-weeks", default=1)
@click.option("--hidden-dim", default=512)
@click.option("--dropout", default=0.1)
@click.option("--n-layers", default=2)
@click.option("--n-heads", default=2) # transformer only
@click.option("--ffnn-layers", default=2)
@click.option("--epochs", default=10)
@click.option("--clip", default=5)
def train(**kwargs):
    globals().update(kwargs)
    wandb.init(config=kwargs, project="drought")
    X_static_train, X_time_train, y_target_train = loadXY(
        "train",
        fuse_past=use_past,
        encode_season=use_season,
        use_prev_year=use_previous_year,
        window_size=window_size,
    )
    print("train shape", X_time_train.shape)
    X_static_valid, X_time_valid, y_target_valid, valid_fips = loadXY(
        "valid",
        fuse_past=use_past,
        return_fips=True,
        encode_season=use_season,
        use_prev_year=use_previous_year,
        window_size=window_size,
    )
    print("validation shape", X_time_valid.shape)
    X_static_train, X_time_train = normalize(X_static_train, X_time_train, fit=True)
    X_static_valid, X_time_valid = normalize(X_static_valid, X_time_valid)
    train_data = TensorDataset(
        torch.tensor(X_time_train),
        torch.tensor(X_static_train),
        torch.tensor(y_target_train[:, :output_weeks]),
    )
    train_loader = DataLoader(
        train_data, shuffle=shuffle, batch_size=batch_size, drop_last=False
    )
    valid_data = TensorDataset(
        torch.tensor(X_time_valid),
        torch.tensor(X_static_valid),
        torch.tensor(y_target_valid[:, :output_weeks]),
    )
    valid_loader = DataLoader(
        valid_data, shuffle=False, batch_size=batch_size, drop_last=False
    )
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("using GPU")
    else:
        device = torch.device("cpu")
        print("using CPU")
    static_dim = 0
    if use_static:
        static_dim = X_static_train.shape[-1]
    if model_type == 'lstm':
        model = DroughtNetLSTM(
            output_weeks,
            X_time_train.shape[-1],
            hidden_dim,
            n_layers,
            ffnn_layers,
            dropout,
            static_dim,
        )
    if model_type == 'transformer':
        model = DroughtNetTransformer(
            output_weeks,
            X_time_train.shape[-1],
            hidden_dim,
            n_layers,
            ffnn_layers,
            dropout,
            static_dim,
            n_heads,
            window_size
        )
    model.to(device)
    loss_function = nn.MSELoss()
    if one_cycle:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    counter = 0
    valid_loss_min = np.Inf
    torch.manual_seed(42)
    np.random.seed(42)
    for i in range(epochs):
        if model_type == 'lstm':
            h = model.init_hidden(batch_size)

        for k, (inputs, static, labels) in tqdm(
            enumerate(train_loader),
            desc=f"epoch {i+1}/{epochs}",
            total=len(train_loader),
        ):
            model.train()
            counter += 1
            if len(inputs) < batch_size and model_type == 'lstm':
                h = model.init_hidden(len(inputs))
            if model_type == 'lstm':
                h = tuple([e.data for e in h])
            inputs, labels, static = (
                inputs.to(device),
                labels.to(device),
                static.to(device),
            )
            model.zero_grad()
            if use_static:
                if model_type == 'lstm':
                    output, h = model(inputs, h, static)
                else:
                    output = model(inputs, static)
            else:
                if model_type == 'lstm':
                    output, h = model(inputs, h)
                else:
                    output = model(inputs)
            loss = loss_function(output, labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if one_cycle:
                scheduler.step()

            with torch.no_grad():
                if k == len(train_loader) - 1 or k == (len(train_loader) - 1) // 2:
                    if model_type == 'lstm':
                        val_h = model.init_hidden(batch_size)
                    val_losses = []
                    model.eval()
                    labels = []
                    preds = []
                    raw_labels = []
                    raw_preds = []
                    for inp, stat, lab in valid_loader:
                        if len(inp) < batch_size and model_type == 'lstm':
                            val_h = model.init_hidden(len(inp))
                        if model_type == 'lstm':
                            val_h = tuple([each.data for each in val_h])
                        inp, lab, stat = inp.to(device), lab.to(device), stat.to(device)
                        if use_static:
                            if model_type == 'lstm':
                                out, val_h = model(inp, val_h, stat)
                            else:
                                out = model(inp, stat)
                        else:
                            if model_type == 'lstm':
                                out, val_h = model(inp, val_h)
                            else:
                                out = model(inp)
                        val_loss = loss_function(out, lab.float())
                        val_losses.append(val_loss.item())
                        for labs in lab:
                            labels.append([int(l.round()) for l in labs])
                            raw_labels.append([float(l) for l in labs])
                        for pred in out:
                            preds.append([int(p.round()) for p in pred])
                            raw_preds.append([float(p) for p in pred])
                    # log data
                    labels = np.array(labels)
                    preds = np.clip(np.array(preds), 0, 5)
                    raw_preds = np.array(raw_preds)
                    raw_labels = np.array(raw_labels)
                    for i in range(output_weeks):
                        log_dict = {
                            "loss": float(loss),
                            "epoch": counter / len(train_loader),
                            "step": counter,
                            "lr": optimizer.param_groups[0]["lr"],
                            "week": i + 1,
                        }
                        # w = f'week_{i+1}_'
                        w = ""
                        log_dict[f"{w}validation_loss"] = np.mean(val_losses)
                        log_dict[f"{w}macro_f1"] = f1_score(
                            labels[:, i], preds[:, i], average="macro"
                        )
                        log_dict[f"{w}micro_f1"] = f1_score(
                            labels[:, i], preds[:, i], average="micro"
                        )
                        log_dict[f"{w}mae"] = mean_absolute_error(
                            raw_labels[:, i], raw_preds[:, i]
                        )
                        wandb.log(log_dict)
                        for j, f1 in enumerate(
                            f1_score(labels[:, i], preds[:, i], average=None)
                        ):
                            log_dict[f"{w}{id2class[j]}_f1"] = f1
                        model.train()
                    if np.mean(val_losses) <= valid_loss_min:
                        torch.save(model.state_dict(), "./state_dict.pt")
                        print(
                            "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                                valid_loss_min, np.mean(val_losses)
                            )
                        )
                        valid_loss_min = np.mean(val_losses)

    # CREATE MAPS AND PREDICTION CSV

    def predict(x, static=None):
        if static is None:
            if model_type == 'lstm':
                out, _ = model(torch.tensor(x), val_h)
            else:
                out = model(torch.tensor(x))
        else:
            if model_type == 'lstm':
                out, _ = model(torch.tensor(x), val_h, static)
            else:
                out = model(torch.tensor(x), static)
        return out

    model.load_state_dict(torch.load("./state_dict.pt"))
    wandb.save("./state_dict.pt")
    model.eval()
    del X_time_train
    X_static_valid, X_time_valid, y_target_valid, valid_fips = loadXY(
        "valid",
        random_state=None,
        fuse_past=use_past,
        return_fips=True,
        encode_season=use_season,
        use_prev_year=use_previous_year,
        window_size=window_size,
    )
    X_static_valid, X_time_valid = normalize(X_static_valid, X_time_valid)
    dict_map = {
        "y_pred": [],
        "y_pred_rounded": [],
        "fips": [],
        "date": [],
        "y_true": [],
        "week": [],
    }
    valid_data = TensorDataset(
        torch.tensor(X_time_valid),
        torch.tensor(X_static_valid),
        torch.tensor(y_target_valid[:, :output_weeks]),
    )
    valid_loader = DataLoader(
        valid_data, shuffle=False, batch_size=batch_size, drop_last=False
    )
    if model_type == 'lstm':
        val_h = tuple([each.data for each in model.init_hidden(batch_size)])
    i = 0
    for x, static, y in tqdm(
        valid_loader,
        desc="validation predictions...",
    ):
        if len(x) != batch_size and model_type == 'lstm':
            val_h = tuple([each.data for each in model.init_hidden(len(x))])
        with torch.no_grad():
            if use_static:
                pred = predict(x, static).clone().detach()
            else:
                pred = predict(x).clone().detach()
        for w in range(output_weeks):
            #            if fips_date[1] not in dict_map['fips']:
            dict_map["y_pred"] += [float(p[w]) for p in pred]
            dict_map["y_pred_rounded"] += [int(p.round()[w]) for p in pred]
            dict_map["fips"] += [f[1][0] for f in valid_fips[i : i + len(x)]]
            dict_map["date"] += [f[1][1] for f in valid_fips[i : i + len(x)]]
            dict_map["y_true"] += [float(item[w]) for item in y]
            dict_map["week"] += [w] * len(x)
        i += len(x)
    df = pd.DataFrame(dict_map)
    df.to_csv("drougths_validation.csv")
    wandb.save("drougths_validation.csv")
    dates = df["date"].unique()
    colorscale = ["#AAAAAA", "#FFFF00", "#FCD37F", "#FFAA00", "#E60000", "#730000"]
    colorscale_diff = [
        [0.0, "rgb(165,0,38)"],
        [0.1111111111111111, "rgb(215,48,39)"],
        [0.2222222222222222, "rgb(244,109,67)"],
        [0.3333333333333333, "rgb(253,174,97)"],
        [0.4444444444444444, "rgb(254,224,144)"],
        [0.5555555555555556, "rgb(224,243,248)"],
        [0.6666666666666666, "rgb(171,217,233)"],
        [0.7777777777777778, "rgb(116,173,209)"],
        [0.8888888888888888, "rgb(69,117,180)"],
        [1.0, "rgb(49,54,149)"],
    ]
    for tgt_date in tqdm(dates, total=len(dates), desc="creating maps..."):
        for w in tqdm(range(output_weeks)):
            tgt_df = df[(df["date"] == tgt_date) & (df["week"] == w)]
            fips = tgt_df["fips"]
            for y_type in ["diff"]:  # y_pred, y_true
                if y_type == "diff":
                    values = (tgt_df["y_pred"] - tgt_df["y_true"]).apply(round)
                    num_cat = len(values.unique())
                    colorscale_temp = []
                    for val in sorted(values.unique()):
                        colorscale_temp.append(colorscale_diff[5 + val][1])
                else:
                    num_cat = len(values.unique())
                    colorscale_temp = colorscale[:num_cat]
                    values = tgt_df[y_type].apply(round)
                fig = ff.create_choropleth(
                    fips=fips,
                    values=values,
                    title=f"{y_type}_{tgt_date}_{w}",
                    colorscale=colorscale_temp,
                )
                fig.layout.template = None
                fig.write_image("images/temp_fig.png")
                im = plt.imread("images/temp_fig.png")
                wandb.log(
                    {
                        f"{y_type}_{tgt_date}_{w}": [
                            wandb.Image(im, caption=f"{y_type} - {tgt_date} - week {w}")
                        ]
                    }
                )


if __name__ == "__main__":
    train()
