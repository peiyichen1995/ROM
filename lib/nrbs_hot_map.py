import torch
from functorch import vmap
import tqdm
import numpy as np
import os
import pdb

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

torch.set_default_dtype(torch.float64)

# bandwidth: n x m
class NRBS(torch.nn.Module):
    def __init__(
        self,
        N,
        n,
        mu,
        m,
        B0,
        neighbour_id,
        neighbour_distance,
        clustering_labels,
    ):
        super(NRBS, self).__init__()
        self.N = N
        self.n = n
        self.mu = mu
        self.m = m
        self.neighbour_id = neighbour_id
        self.neighbour_distance = neighbour_distance
        self.clustering_labels = clustering_labels

        self.encoder1 = torch.nn.Linear(self.N, 200)
        self.encoder2 = torch.nn.Linear(200, self.n)

        self.decoder = torch.nn.Linear(self.n, self.N)
        hotness_map = []
        in_dim = self.n
        for dim in self.m:
            hotness_map.append(torch.nn.Linear(in_dim, dim))
            in_dim = dim
        self.hotness_map = torch.nn.ModuleList(hotness_map)

        self.B = torch.nn.Parameter(torch.tensor([B0]))

    def hotness(self, x):
        for i in range(len(self.m) - 1):
            x = self.hotness_map[i](x)
            x = x * torch.sigmoid(x)
        x = self.hotness_map[-1](x)
        # x = torch.sigmoid(x)
        x = 1 / (1 + torch.exp(-x * 0.01))
        return x

    def encode(self, x):
        x = self.encoder1(x)
        x = torch.sigmoid(x)
        x = self.encoder2(x)
        return x

    # distance: N x mu
    # w: n x N ([0 element_size])
    # mu: number of neighbour elements
    def bubble(self, distance, w, mu):
        window = torch.relu(
            -(distance.unsqueeze(0).expand(self.n, -1, -1) ** 2)
            / (w.unsqueeze(-1) * mu) ** 2
            + 1
        )
        window = window / torch.sum(window, dim=2, keepdim=True)
        # n x N x mu
        return window

    # x: n x N
    # neighbour_id: N x mu
    # w: n x N
    def convolve(self, x, neighbour_id, neighbour_distance, w, mu):
        # n x N x mu
        # 1.3GB
        bubbles = self.bubble(neighbour_distance, w, mu)
        return torch.sum(x[:, neighbour_id] * bubbles, dim=-1)

    def decode(self, encoded):
        # return self.linear_decoder(encoded)
        b = encoded.shape[0]

        convolved_basis = torch.empty((b, self.n, self.N), device=encoded.device)

        # b x m
        for i in range(b):
            H = (
                self.hotness(encoded[i])[self.clustering_labels]
                .unsqueeze(0)
                .expand(self.n, -1)
            )
            # n x N
            bandwidths = (1 - H / 2) ** torch.arange(
                self.n, device=encoded.device
            ).unsqueeze(-1) * self.B
            convolved_basis[i] = self.convolve(
                self.decoder.weight.T,
                self.neighbour_id,
                self.neighbour_distance,
                bandwidths,
                self.mu,
            )

        # batch size x N
        return torch.bmm(encoded.unsqueeze(1), convolved_basis).squeeze(
            1
        ) + self.decoder.bias.unsqueeze(0)

    def forward(self, x):
        return self.decode(self.encode(x))


class EncoderDecoder(torch.nn.Module):
    def __init__(
        self,
        N,
        n,
        mu,
        m,
        B0,
        neighbour_id,
        neighbour_distance,
        clustering_labels,
        device,
    ):
        super(EncoderDecoder, self).__init__()
        self.nrbs = NRBS(
            N=N,
            n=n,
            mu=mu,
            m=m,
            B0=B0,
            neighbour_id=neighbour_id.to(device),
            neighbour_distance=neighbour_distance.to(device),
            clustering_labels=clustering_labels.to(device),
        ).to(device)
        self.device = device

    def train(self, train_data_loader, effective_batch=64, epochs=1):
        writer = SummaryWriter()
        loss_func = torch.nn.MSELoss(reduction="sum")
        model_name = "models/n_m_hot_map_200_200.pth"

        # # L-BFGS
        # def closure():
        #     objective = 0
        #     lbfgs.zero_grad()
        #     for u, u_dot in train_data_loader:
        #         approximates = self.nrbs(u)
        #         loss = loss_func(u_dot, approximates)
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

        lr = 1e-3
        optim = torch.optim.Adam(self.nrbs.parameters(), lr=lr)

        accu_itr = effective_batch // train_data_loader.batch_size

        best_loss = 0
        with torch.no_grad():
            for u in tqdm.tqdm(train_data_loader):
                approximates = self.nrbs(u)
                loss = loss_func(u, approximates)
                best_loss = best_loss + loss.item()

        print("Initial loss = {:}".format(best_loss / len(train_data_loader)))

        patience = 0
        for i in range(epochs):
            curr_loss = 0
            for j, u in enumerate(tqdm.tqdm(train_data_loader)):

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
            writer.add_scalar("loss/current", curr_loss / len(train_data_loader), i)
            writer.add_scalar("loss/best", best_loss / len(train_data_loader), i)
            writer.add_scalar("lr", lr, i)

    def forward(self, x):
        return self.nrbs(x)
