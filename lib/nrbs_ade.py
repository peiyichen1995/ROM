import torch
from functorch import vmap
import tqdm
import numpy as np
import os
import pdb

from torch.utils.tensorboard import SummaryWriter

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

        torch.nn.init.kaiming_normal_(self.encoder1.weight)
        torch.nn.init.kaiming_normal_(self.encoder2.weight)
        torch.nn.init.kaiming_normal_(self.decoder1.weight)
        torch.nn.init.kaiming_normal_(self.decoder2.weight)

    def encode(self, x):
        x = self.encoder1(x)
        x = torch.sigmoid(x)
        x = self.encoder2(x)
        return x

    def decode(self, x):
        x = self.decoder1(x)
        x = torch.sigmoid(x)
        x = self.decoder2(x)
        return x

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

    def eval(self, data_loader, norm, Nt):

        loss = 0
        mse = 0
        with torch.no_grad():
            for u in data_loader:
                approximates = self.nrbs(u)
                loss = loss + torch.sum(
                    torch.sum((u - approximates) ** 2, dim=1) / torch.sum(u**2, dim=1)
                )
                mse = mse + torch.sum((u - approximates) ** 2)

            torch.cuda.empty_cache()
        return (mse / 3600 / Nt, torch.sqrt(loss / Nt))

    def train(
        self,
        train_dataloader,
        test_dataloader,
        unseen_dataloader,
        train_norm,
        test_norm,
        unseen_norm,
        comment,
        epochs=1,
    ):

        writer = SummaryWriter(comment=comment)

        loss_func = torch.nn.MSELoss(reduction="sum")
        model_name = "models/ade.pth"

        lr = 1e-3
        optim = torch.optim.Adam(self.nrbs.parameters(), lr=lr)

        _, best_unseen_proj_err = self.eval(unseen_dataloader, unseen_norm, 1501)

        writer.add_scalar("projection_err/unseen_best", best_unseen_proj_err, -1)

        patience = 0
        for i in range(epochs):
            curr_loss = 0
            for u in train_dataloader:

                approximates = self.nrbs(u)
                objective = loss_func(u, approximates)
                objective.backward()
                optim.step()
                optim.zero_grad()

                torch.cuda.empty_cache()

            train_loss, train_proj_err = self.eval(train_dataloader, train_norm, 5404)
            test_loss, test_proj_err = self.eval(test_dataloader, test_norm, 600)
            unseen_loss, unseen_proj_err = self.eval(
                unseen_dataloader, unseen_norm, 1501
            )

            if unseen_proj_err < best_unseen_proj_err:
                patience = 0
                best_unseen_proj_err = unseen_proj_err
                if os.path.isfile(model_name):
                    os.remove(model_name)
                torch.save(self.nrbs, model_name)
            else:
                patience = patience + 1
            if patience == 10:
                self.nrbs = torch.load(model_name)
                patience = 0
                lr = lr / 10
                optim = torch.optim.Adam(self.nrbs.parameters(), lr=lr)

            writer.add_scalar("loss/train_current", train_loss, i)
            writer.add_scalar("loss/test_current", test_loss, i)
            writer.add_scalar("loss/unseen_current", unseen_loss, i)

            writer.add_scalar("projection_err/train_current", train_proj_err, i)
            writer.add_scalar("projection_err/test_current", test_proj_err, i)
            writer.add_scalar("projection_err/unseen_current", unseen_proj_err, i)
            writer.add_scalar("projection_err/unseen_best", best_unseen_proj_err, i)

            writer.add_scalar("lr", lr, i)

    def forward(self, x):
        return self.nrbs(x)
