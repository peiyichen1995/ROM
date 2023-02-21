import torch
import tqdm
import numpy as np
import os
from torch.utils.data import DataLoader
import pdb

torch.set_default_dtype(torch.float64)

# bandwidth: n x m
class NRBS(torch.nn.Module):
    def __init__(self, N, n):
        super(NRBS, self).__init__()
        self.N = N
        self.n = n

        self.encoder = torch.nn.Linear(self.N, self.n)
        self.decoder = torch.nn.Linear(self.n, self.N)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, encoded):
        return self.decoder(encoded)

    def forward(self, x):
        return self.decode(self.encode(x))


class EncoderDecoder(torch.nn.Module):
    def __init__(self, N, n, device):
        super(EncoderDecoder, self).__init__()
        self.nrbs = NRBS(N=N, n=n).to(device)
        self.device = device

    def train(self, train_data_loader, epochs=1):

        loss_func = torch.nn.MSELoss(reduction="none")
        best_loss = float("inf")

        # L-BFGS
        def closure():
            lbfgs.zero_grad()
            approximates = self.nrbs(x)
            objective = torch.sum(loss_func(x, approximates))
            objective.backward(retain_graph=True)
            return objective

        lbfgs = torch.optim.LBFGS(
            self.nrbs.parameters(),
            history_size=10,
            max_iter=4,
            line_search_fn="strong_wolfe",
            lr=1,
        )

        for i in range(epochs):
            self.nrbs.train()
            curr_loss = 0
            for x in tqdm.tqdm(train_data_loader):
                approximates = self.nrbs(x)
                lbfgs.zero_grad()
                objective = torch.sum(loss_func(x, approximates))
                objective.backward(retain_graph=True)
                lbfgs.step(closure)
            self.nrbs.eval()
            for x in tqdm.tqdm(train_data_loader):
                x = x.to(self.device)
                approximates = self.nrbs(x)
                objective = torch.sum(loss_func(x, approximates))
                curr_loss = curr_loss + objective.item()
            print("Itr {:}, loss = {:}".format(i, curr_loss))
            if curr_loss < best_loss:
                if os.path.isfile("models/nrbs_n_m.pth"):
                    os.remove("models/nrbs_n_m.pth")
                torch.save(self.nrbs, "models/nrbs_n_m.pth")

    def forward(self, x):
        return self.nrbs(x)


n = 10
N = 121 * 121
datas = torch.rand(10, N)

device = torch.device("cuda")

ed = EncoderDecoder(N, n, device)

batch_size = 1
datas = datas.to(device)
train_data = DataLoader(datas, batch_size=batch_size, shuffle=True)


ed.train(train_data_loader=train_data, epochs=1)
