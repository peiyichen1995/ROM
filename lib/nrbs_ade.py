import torch
from functorch import vmap
import tqdm
import numpy as np
import os
import pdb

torch.set_default_dtype(torch.float64)

# bandwidth: n x m
class NRBS(torch.nn.Module):
    def __init__(self, N, n):
        super(NRBS, self).__init__()
        self.N = N
        self.n = n

        self.encoder1 = torch.nn.Linear(self.N, 6728)
        self.encoder2 = torch.nn.Linear(6728, self.n)

        self.decoder1 = torch.nn.Linear(self.n, 33730)
        self.decoder2 = torch.nn.Linear(33730, self.N)

        torch.nn.init.kaiming_normal_(self.encoder1.weight, mode="fan_out")
        torch.nn.init.kaiming_normal_(self.encoder2.weight, mode="fan_out")
        torch.nn.init.kaiming_normal_(self.decoder1.weight, mode="fan_out")
        torch.nn.init.kaiming_normal_(self.decoder2.weight, mode="fan_out")

    def encode(self, x):
        x = self.encoder1(x)
        x = torch.sigmoid(x)
        x = self.encoder2(x)
        return x

    def decode(self, encoded):
        encoded = self.decoder1(encoded)
        encoded = self.decoder2(encoded)
        return encoded

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
        self.nrbs = NRBS(
            N=N,
            n=n,
        ).to(device)
        self.device = device

    def train(self, train_data_loader, effective_batch=64, epochs=1):

        loss_func = torch.nn.MSELoss(reduction="sum")
        model_name = "models/ade.pth"

        # # L-BFGS
        # def closure():
        #     objective = 0
        #     lbfgs.zero_grad()
        #     for u in train_data_loader:
        #         approximates = self.nrbs(u)
        #         loss = loss_func(u, approximates)
        #         loss.backward()
        #         objective = objective + loss
        #     return objective

        # lbfgs = torch.optim.LBFGS(
        #     self.nrbs.parameters(),
        #     history_size=20,
        #     max_iter=10,
        #     line_search_fn="strong_wolfe",
        #     lr=1,
        # )

        lr = 1e-4
        optim = torch.optim.Adam(self.nrbs.parameters(), lr=lr)

        accu_itr = effective_batch // train_data_loader.batch_size

        best_loss = 0
        with torch.no_grad():
            for u in tqdm.tqdm(train_data_loader):
                approximates = self.nrbs(u)
                loss = loss_func(u, approximates)
                best_loss = best_loss + loss.item()

        print(
            "Initial loss = {:}".format(
                best_loss / len(train_data_loader) / train_data_loader.batch_size
            )
        )

        patience = 0
        for i in range(epochs):
            curr_loss = 0
            for j, (u) in enumerate(tqdm.tqdm(train_data_loader)):

                approximates = self.nrbs(u)
                objective = loss_func(u, approximates)
                objective.backward()

                if ((j + 1) % accu_itr == 0) or (j + 1 == len(train_data_loader)):
                    optim.step()
                    optim.zero_grad()
                    # lbfgs.step(closure)
                    # lbfgs.zero_grad()
                torch.cuda.empty_cache()

            with torch.no_grad():
                for u in tqdm.tqdm(train_data_loader):
                    approximates = self.nrbs(u)
                    loss = loss_func(u, approximates)
                    curr_loss = curr_loss + loss.item()

            if curr_loss < best_loss:
                patience = 0
                best_loss = curr_loss
                if os.path.isfile(model_name):
                    os.remove(model_name)
                torch.save(self.nrbs, model_name)
            else:
                patience = patience + 1
            if patience == 10:
                patience = 0
                lr = lr / 10
                optim = torch.optim.Adam(self.nrbs.parameters(), lr=lr)
            print(
                "Itr {:}, curr_loss = {:}, best_loss = {:}, lr = {:}".format(
                    i,
                    curr_loss / len(train_data_loader) / train_data_loader.batch_size,
                    best_loss / len(train_data_loader) / train_data_loader.batch_size,
                    lr,
                )
            )

    def forward(self, x):
        return self.nrbs(x)
