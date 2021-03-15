import datetime
import sys
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import seaborn as sns

sys.path.append("./ts_transformer")

from tst import Transformer
from tst.loss import OZELoss

from src.dataset import OzeDataset
from src.utils import compute_loss
from src.visualization import (
    map_plot_function,
    plot_values_distribution,
    plot_error_distribution,
    plot_errors_threshold,
    plot_visual_sample,
)

class DroughtTransformer:

    def __init__(
        self,
        BATCH_SIZE,
        LR,
        EPOCHS,
        dropout,
        d_model,
        d_input,
        d_output,
        ALPHA
    ):

        # Training parameters
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_WORKERS = 0
        self.EPOCHS = EPOCHS

        # Model parameters
        self.q = 8  # Query size
        self.v = 8  # Value size
        self.h = 8  # Number of heads
        self.N = 4  # Number of encoder and decoder to stack
        self.attention_size = 12  # Attention window size
        self.pe = None  # Positional encoding
        self.chunk_mode = None

        # Config
        sns.set()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.load_network(d_model, d_input, d_output, ALPHA, LR, dropout)

    def load_network(d_model, d_input, d_output, ALPHA, LR, dropout):
        # Load transformer with Adam optimizer and MSE loss function
        self.net = Transformer(
            d_input,
            d_model,
            d_output,
            q,
            v,
            h,
            N,
            attention_size=attention_size,
            dropout=dropout,
            chunk_mode=chunk_mode,
            pe=pe,
        ).to(device)
        
        sef.load_network(d_model, d_input, d_output, ALPHA, LR, dropout)

    def load_network(d_model, d_input, d_output, ALPHA, LR, dropout):
        # Load transformer with Adam optimizer and MSE loss function
        self.net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
        self.optimizer = optim.Adam(net.parameters(), lr=LR)
        self.loss_function = OZELoss(alpha=ALPHA)

        self.net = self.net.float()

    def get_model(self):
        return self.net

    def get_loss(self):
        return self.loss_function

    def get_optimizer(self):
        return self.optimizer

    def train(self, dataloader_train, dataloader_val, model_save_path):

        val_loss_best = np.inf

        for idx_epoch in range(self.EPOCHS):
            running_loss = 0
            with tqdm(
                total=len(dataloader_train.dataset),
                desc=f"[Epoch {idx_epoch+1:3d}/{self.EPOCHS}]",
            ) as pbar:
            with tqdm(total=len(dataloader_train.dataset), desc=f"[Epoch {idx_epoch+1:3d}/{self.EPOCHS}]") as pbar:

                for idx_batch, (x, y) in enumerate(dataloader_train):
                    self.optimizer.zero_grad()
                    # Propagate input
                    netout = self.net(x.float().to(device))
                    y = y[:, :, np.newaxis]
                    y = y[:,:,np.newaxis]
                    # Comupte loss
                    loss = self.loss_function(y.float().to(device), netout)
                    # Backpropage loss
                    loss.backward()
                    # Update weights
                    self.optimizer.step()
                    running_loss += loss.item()
                    pbar.set_postfix({"loss": running_loss / (idx_batch + 1)})
                    pbar.update(x.shape[0])

                train_loss = running_loss / len(dataloader_train)
                val_loss = src.utils.compute_loss(
                    self.net, dataloader_val, self.loss_function, self.device
                ).item()
                pbar.set_postfix({"loss": train_loss, "val_loss": val_loss})

                print("COMPARING LOSS")
                if val_loss < val_loss_best:
                    val_loss_best = val_loss
                    print("Saving")
                    pbar.set_postfix({'loss': running_loss/(idx_batch+1)})
                    pbar.update(x.shape[0])


                train_loss = running_loss/len(dataloader_train)
                val_loss = src.utils.compute_loss(self.net, dataloader_val, self.loss_function, self.device).item()
                pbar.set_postfix({'loss': train_loss, 'val_loss': val_loss})

                print ("COMPARING LOSS")
                if val_loss < val_loss_best:
                    val_loss_best = val_loss
                    print ("Saving")
                    torch.save(net.state_dict(), model_save_path)

        print(f"model exported to {model_save_path} with loss {val_loss_best:5f}")
