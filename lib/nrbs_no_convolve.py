import torch
from functorch import vmap
import tqdm
import numpy as np
import os

torch.set_default_dtype(torch.float64)

# bandwidth: n x m
class NRBS(torch.nn.Module):
    def __init__(self, N, n):
        super(NRBS, self).__init__()
        self.N = N
        self.n = n

        torch.manual_seed(0)

        self.encoder1 = torch.nn.Linear(self.N, 6728)
        self.encoder2 = torch.nn.Linear(6728, self.n)

        self.decoder1 = torch.nn.Linear(self.n, 3373)
        self.decoder2 = torch.nn.Linear(3373, self.N)

    def encode(self, x):
        x = self.encoder1(x)
        x = x * torch.sigmoid(x)
        return self.encoder2(x)

    def decode(self, encoded):
        return self.decoder2(self.decoder1(encoded))

    def forward(self, x):
        return self.decode(self.encode(x))


class EncoderDecoder(torch.nn.Module):
    def __init__(
        self,
        N,
        n,
        device,
    ):
        super(EncoderDecoder, self).__init__()
        self.nrbs = NRBS(N=N, n=n).to(device)
        self.device = device

    def train(self, train_data_loader, effective_batch=240, epochs=1):

        optim = torch.optim.Adam(self.nrbs.parameters(), 1e-5)
        loss_func = torch.nn.MSELoss(reduction="sum")
        best_loss = float("inf")
        self.nrbs.train()
        accu_itr = effective_batch // train_data_loader.batch_size

        for i in range(epochs):
            curr_loss = 0

            for j, x in enumerate(tqdm.tqdm(train_data_loader)):
                approximates = self.nrbs(x[:, : self.nrbs.N])
                loss = loss_func(x[:, self.nrbs.N :], approximates)
                loss.backward()
                # if ((j + 1) % accu_itr == 0) or (j + 1 == len(train_data_loader)):
                optim.step()
                optim.zero_grad()
                torch.cuda.empty_cache()
            with torch.no_grad():
                for x in tqdm.tqdm(train_data_loader):
                    approximates = self.nrbs(x[:, : self.nrbs.N])
                    loss = loss_func(x[:, self.nrbs.N :], approximates)
                    curr_loss = curr_loss + loss.item()
            print("Itr {:}, loss = {:}".format(i, curr_loss / 1000))
            if curr_loss < best_loss:
                best_loss = curr_loss
                if os.path.isfile("models/nrbs_n_m_test.pth"):
                    os.remove("models/nrbs_n_m_test.pth")
                torch.save(self.nrbs, "models/nrbs_n_m_test.pth")

    def forward(self, x):
        return self.nrbs(x)
