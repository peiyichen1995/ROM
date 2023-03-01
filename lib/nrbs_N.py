import torch
from functorch import vmap
import tqdm
import numpy as np
import os
import pdb

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

        torch.manual_seed(0)

        self.encoder = torch.nn.Linear(self.N, self.n)
        # self.linear_decoder = torch.nn.Linear(self.n, self.N, bias=False)
        self.decoder = torch.nn.Linear(self.N, self.n, bias=False)
        self.bandwidth_layers = torch.nn.Linear(self.n, self.m, bias=False)

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
            # N
            bandwidths = bandwidths[self.clustering_labels]
            # n x N
            bandwidths = bandwidths.unsqueeze(0).expand(self.n, -1)

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

    def train(self, train_data_loader, effective_batch=64, epochs=1):

        loss_func = torch.nn.MSELoss(reduction="sum")
        best_loss = float("inf")

        # # L-BFGS
        # def closure():
        #     lbfgs.zero_grad()
        #     approximates = self.nrbs(x[:, : self.nrbs.N])
        #     objective = loss_func(x[:, self.nrbs.N :], approximates)
        #     objective.backward()
        #     return objective

        # lbfgs = torch.optim.LBFGS(
        #     self.nrbs.parameters(),
        #     history_size=20,
        #     max_iter=10,
        #     line_search_fn="strong_wolfe",
        #     lr=1,
        # )

        optim = torch.optim.Adam(self.nrbs.parameters(), 1e-4)

        accu_itr = effective_batch // train_data_loader.batch_size

        curr_loss = 0
        with torch.no_grad():
            for x in tqdm.tqdm(train_data_loader):
                approximates = self.nrbs(x[:, : self.nrbs.N])
                loss = loss_func(x[:, self.nrbs.N :], approximates)
                curr_loss = curr_loss + loss.item()

        print("Loss = {:}".format(curr_loss / 1000))

        for i in range(epochs):
            curr_loss = 0
            for j, x in enumerate(tqdm.tqdm(train_data_loader)):
                # lbfgs.zero_grad()
                approximates = self.nrbs(x[:, : self.nrbs.N])
                objective = loss_func(x[:, self.nrbs.N :], approximates)
                objective.backward()
                # lbfgs.step(closure)

                if ((j + 1) % accu_itr == 0) or (j + 1 == len(train_data_loader)):
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
                if os.path.isfile("models/nrbs_N.pth"):
                    os.remove("models/nrbs_N.pth")
                torch.save(self.nrbs, "models/nrbs_N.pth")

    def forward(self, x):
        return self.nrbs(x)
