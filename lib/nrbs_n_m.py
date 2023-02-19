import torch
from functorch import vmap
import tqdm
import numpy as np
import os

torch.set_default_dtype(torch.float64)

# bandwidth: n x m
class NRBS(torch.nn.Module):
    def __init__(self, N, n, mu, m, neighbours, group_indices, device):
        super(NRBS, self).__init__()
        self.N = N
        self.n = n
        self.mu = mu
        self.m = m
        self.neighbours = neighbours.to(device)
        self.group_indices = group_indices

        self.encoder = torch.nn.Linear(self.N, self.n)
        self.decoder = torch.nn.Linear(self.n, self.N)
        self.bandwidth_layers = torch.nn.Linear(self.n, self.n * self.m)

    def encode(self, x):
        return self.encoder(x)

    def get_group_idx_smoothed_basis(self, groud_idx, basises, bubbles):
        curr_bubble = bubbles[:, :, groud_idx, :]
        curr_basis = basises[:, self.group_indices[groud_idx], :]

        # n x mu x N/m
        curr_basis = curr_basis.permute((0, 2, 1))

        # n x batch x mu
        curr_bubble = curr_bubble.permute((1, 0, 2))

        # n x b x N/m
        return torch.bmm(curr_bubble, curr_basis)

    def decode(self, encoded):
        batch_size = encoded.shape[0]
        vmap_bubble = vmap(self.bubble, in_dims=0)
        vmap_vmap_bubble = vmap(vmap_bubble, in_dims=0)
        vmap_vmap_vmap_bubble = vmap(vmap_vmap_bubble, in_dims=0)

        # batch size x n x m
        bandwidths = torch.bmm(
            encoded.repeat(self.n, 1, 1), self.bandwidth_layers
        ).permute(1, 0, 2)
        bandwidths = torch.sigmoid(bandwidths)
        # batch size x n x m x mu
        bubbles = vmap_vmap_vmap_bubble(bandwidths)

        node_idxs = torch.linspace(0, self.N - 1, self.N, dtype=torch.long)
        basises = self.decoder[:, self.getNeighbours(node_idxs)]

        # n x batch_size x N
        smoothed_basis = torch.zeros((self.n, batch_size, self.N), device=self.device)
        for i in range(len(self.group_indices)):
            # n x b x N/m
            smoothed_basis[
                :, :, self.group_indices[i]
            ] = self.get_group_idx_smoothed_basis(i, basises, bubbles)

        # batch x n x N
        smoothed_basis = smoothed_basis.permute((1, 0, 2))

        # # batch size x n x N
        # smoothed_basis = self.smooth_basis(bubbles=bubbles)

        # batch size x 1 x n
        encoded = encoded.unsqueeze(2).permute((0, 2, 1))
        # batch size x N
        return torch.bmm(encoded, smoothed_basis).squeeze(1)

    def forward(self, x):
        return self.decode(self.encode(x))

    def bubble(self, w):
        x = torch.arange(self.mu, device=self.device)
        window = torch.relu(-(x**2) / (w * self.mu) ** 2 + 1)
        window = window / torch.sum(window)
        return window

    def getNeighbours(self, idx):
        return self.neighbours[idx]

    def convolve(self, x, bubble):
        return x * bubble

    def smooth_basis(self, bubbles):
        node_idxs = torch.linspace(0, self.N - 1, self.N, dtype=torch.long)
        return torch.stack(
            [
                self.smooth_vec(i, node_idx=node_idxs, bubbles=bubbles)
                for i in range(self.n)
            ],
            dim=1,
        ).squeeze(2)

    def smooth_vec(self, basis_idx, node_idx, bubbles):
        return (self.decoder[basis_idx][self.getNeighbours(node_idx)] * bubbles).sum(
            dim=2
        )

    def smooth(self, basis_idx, node_idx, bubbles):
        return torch.sum(
            self.decoder[basis_idx][self.getNeighbours(node_idx)]
            * bubbles[basis_idx][node_idx]
        )


class EncoderDecoder(torch.nn.Module):
    def __init__(self, N, n, mu, m, neighbours, group_indices, device):
        super(EncoderDecoder, self).__init__()
        self.nrbs = NRBS(
            N=N,
            n=n,
            mu=mu,
            m=m,
            neighbours=neighbours,
            group_indices=group_indices,
            device=device,
        ).to(device)
        self.device = device

    def train(self, train_data_loader, effective_batch=100, epochs=1):

        loss_func = torch.nn.MSELoss(reduction="none")
        best_loss = float("inf")

        # L-BFGS
        def closure():
            lbfgs.zero_grad()
            objective = torch.sum(loss_func(x, approximates))
            objective.backward(retain_graph=True)
            return objective

        lbfgs = torch.optim.LBFGS(
            self.nrbs.parameters(),
            history_size=10,
            max_iter=4,
            line_search_fn="strong_wolfe",
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
