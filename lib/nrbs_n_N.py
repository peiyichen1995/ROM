import torch
from functorch import vmap
import tqdm
import numpy as np
import os

torch.set_default_dtype(torch.float64)

# bandwidth: 1 x N
class NRBS(torch.nn.Module):
    def __init__(self, N, n, mu, neighbours, device):
        super(NRBS, self).__init__()
        self.N = N
        self.n = n
        self.mu = mu
        self.neighbours = neighbours.to(device)
        self.device = device

        self.encoder = torch.nn.Linear(self.N, self.n, device=self.device)
        # self.decoder = torch.nn.Linear(self.n, self.N)
        self.decoder = torch.nn.Parameter(
            torch.Tensor(self.n, self.N).uniform_(-0.01, 0.01), requires_grad=True
        )

        # self.bandwidth_layers = [
        #     torch.nn.Linear(self.n, self.N, device=self.device) for i in range(self.n)
        # ]

        self.bandwidth_layers = torch.nn.Parameter(
            torch.Tensor(self.n, self.n, self.N).uniform_(-0.01, 0.01),
            requires_grad=True,
        )

    def get_bandwidth(self, encoded):
        return torch.stack(
            [torch.sigmoid(self.bandwidth_layers[i](encoded)) for i in range(self.n)],
            dim=1,
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, encoded):
        vmap_bubble = vmap(self.bubble, in_dims=0)
        vmap_vmap_bubble = vmap(vmap_bubble, in_dims=0)
        vmap_vmap_vmap_bubble = vmap(vmap_vmap_bubble, in_dims=0)
        # batch size x N x mu
        # bubbles = vmap_vmap_bubble(self.get_bandwidth(encoded))
        # bubbles = vmap_vmap_vmap_bubble(self.get_bandwidth(encoded))

        # batch size x n x N
        bandwidths = torch.bmm(
            encoded.repeat(self.n, 1, 1), self.bandwidth_layers
        ).permute(1, 0, 2)
        bandwidths = torch.sigmoid(bandwidths)
        # batch size x n x N x mu
        bubbles = vmap_vmap_vmap_bubble(bandwidths)

        # print("bubbles shape: ", bubbles.shape)
        # batch size x n x N
        smoothed_basis = self.smooth_basis(bubbles=bubbles)
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
        return (
            self.decoder[basis_idx][self.getNeighbours(node_idx)]
            * bubbles[:, basis_idx]
        ).sum(dim=2)

    def smooth(self, basis_idx, node_idx, bubbles):
        return torch.sum(
            self.decoder[basis_idx][self.getNeighbours(node_idx)]
            * bubbles[basis_idx][node_idx]
        )


class EncoderDecoder(torch.nn.Module):
    def __init__(self, N, n, mu, neighbours, device):
        super(EncoderDecoder, self).__init__()
        self.nrbs = NRBS(N=N, n=n, mu=mu, neighbours=neighbours, device=device).to(
            device
        )
        self.device = device

    def train(self, train_data_loader, epochs=1):

        optim = torch.optim.AdamW(self.nrbs.parameters(), 1e-6)
        loss_func = torch.nn.MSELoss(reduction="none")
        best_loss = float("inf")
        for i in range(epochs):
            self.nrbs.train()
            curr_loss = 0
            for x in tqdm.tqdm(train_data_loader):
                x = x.to(self.device)
                approximates = self.nrbs(x)
                # loss = torch.sum(loss_func(x, approximates), dim=1)
                loss = torch.sum(loss_func(x, approximates))
                curr_loss = curr_loss + loss.item()
                loss.backward()
                optim.step()
                optim.zero_grad()

            print("Itr {:}, loss = {:}".format(i, curr_loss))
            if curr_loss < best_loss:
                if os.path.isfile("models/nrbs_n_N.pth"):
                    os.remove("models/nrbs_n_N.pth")
                torch.save(self.nrbs, "models/nrbs_n_N.pth")

    def forward(self, x):
        return self.nrbs(x)
