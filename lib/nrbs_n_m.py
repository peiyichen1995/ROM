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
    def __init__(
        self, N, n, mu, m, neighbour_id, neighbour_distance, clustering_labels
    ):
        super(NRBS, self).__init__()
        self.N = N
        self.n = n
        self.mu = mu
        self.m = m
        self.neighbour_id = neighbour_id
        self.neighbour_distance = neighbour_distance
        self.clustering_labels = clustering_labels

        self.encoder = torch.nn.Linear(self.N, self.n)

        self.decoder = torch.nn.Linear(self.N, self.n, bias=False)
        self.bandwidth_layers = torch.nn.Linear(self.n, self.n * self.m)

    def encode(self, x):
        return self.encoder(x)

    # distance: N x mu
    # w: n x N ([0 element_size])
    # mu: number of neighbour elements
    def bubble(self, distance, w, mu):
        n = w.shape[0]
        window = torch.relu(
            -(distance.unsqueeze(0).expand(n, -1, -1) ** 2)
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

        for i in range(b):
            bandwidths = torch.sigmoid(self.bandwidth_layers(encoded[i]))
            bandwidths = (1 / 60 - 2 / 60 / self.mu) * bandwidths + 2 / 60 / self.mu
            bandwidths = bandwidths.reshape(self.n, self.m)
            bandwidths = bandwidths[:, self.clustering_labels]

            convolved_basis[i] = self.convolve(
                self.decoder.weight,
                self.neighbour_id,
                self.neighbour_distance,
                bandwidths,
                self.mu,
            )

        # batch size x N
        return torch.bmm(encoded.unsqueeze(1), convolved_basis).squeeze(1)

    def forward(self, x):
        return self.decode(self.encode(x))


class EncoderDecoder(torch.nn.Module):
    def __init__(
        self, N, n, mu, m, neighbour_id, neighbour_distance, clustering_labels, device
    ):
        super(EncoderDecoder, self).__init__()
        self.nrbs = NRBS(
            N=N,
            n=n,
            mu=mu,
            m=m,
            neighbour_id=neighbour_id.to(device),
            neighbour_distance=neighbour_distance.to(device),
            clustering_labels=clustering_labels.to(device),
        ).to(device)
        self.device = device

    def train(
        self, train_dataloader, test_dataloader, x_ref, effective_batch=64, epochs=1
    ):

        optimizer = torch.optim.Adam(self.nrbs.parameters(), lr=1e-3, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.1, patience=10
        )

        loss_func = torch.nn.MSELoss()

        best_loss = 1000
        writer = SummaryWriter()
        for epoch in range(epochs):
            for x in tqdm.tqdm(train_dataloader):
                optimizer.zero_grad()
                x_tilde = self.nrbs(x - x_ref) + x_ref
                l = torch.sqrt(torch.sum((x_tilde - x) ** 2))
                l.backward()
                optimizer.step()

            l_train = 0
            l_test = 0
            with torch.no_grad():
                for x in tqdm.tqdm(train_dataloader):
                    X_tilde = self.nrbs(x - x_ref)
                    l_train = l_train + torch.sum((x - X_tilde) ** 2)
                for x in tqdm.tqdm(test_dataloader):
                    X_tilde = self.nrbs(x - x_ref)
                    l_test = l_test + torch.sum((x - X_tilde) ** 2)

                l_test = torch.sqrt(l_test) / torch.sqrt(
                    torch.sum(test_dataloader.dataset**2)
                )
                l_train = torch.sqrt(l_train) / torch.sqrt(
                    torch.sum(train_dataloader.dataset**2)
                )
                l_train_mse = l_train / 3600 / len(train_dataloader.dataset)

            scheduler.step(l_train)

            writer.add_scalar("loss/train", l_train, epoch)
            writer.add_scalar("loss/test", l_test, epoch)
            writer.add_scalar("loss/mse_train", l_train_mse, epoch)

            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

    def forward(self, x):
        return self.nrbs(x)
