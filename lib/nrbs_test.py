import torch
from functorch import vmap
import tqdm
import numpy as np
import os

torch.set_default_dtype(torch.float64)

# bandwidth: n x m
class NRBS(torch.nn.Module):
    def __init__(
        self,
        N,
        n,
        mu,
        m,
        neighbour_id,
        neighbour_distance,
        clustering_labels,
        group_indices,
        device,
    ):
        super(NRBS, self).__init__()
        self.N = N
        self.n = n
        self.mu = mu
        self.m = m
        self.neighbour_id = neighbour_id.to(device)
        self.neighbour_distance = neighbour_distance.to(device)
        self.clustering_labels = clustering_labels.to(device)
        self.group_indices = group_indices
        self.device = device

        torch.manual_seed(0)

        self.encoder = torch.nn.Linear(self.N, self.n, bias=False)
        # self.decoder = torch.nn.Linear(self.n, self.N)
        # self.decoder = torch.nn.Parameter(
        #     torch.Tensor(self.n, self.N).uniform_(-0.01, 0.01), requires_grad=True
        # )

        self.decoder = torch.nn.Linear(self.N, self.n, bias=False)

        # self.bandwidth_layers = torch.nn.Parameter(
        #     torch.Tensor(self.n, self.n * self.m).uniform_(-0.01, 0.01),
        #     requires_grad=True,
        # )

        self.bandwidth_layers = torch.nn.Linear(self.n, self.n * self.m, bias=False)

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

        # b x n x m
        bandwidths = torch.sigmoid(self.bandwidth_layers(encoded))
        bandwidths = (1 / 60 - 4 / 60 / self.mu) * bandwidths + 4 / 60 / self.mu
        bandwidths = bandwidths.reshape(-1, self.n, self.m)
        # b x n x N
        bandwidths = bandwidths[:, :, self.clustering_labels]

        # n x N x mu
        basises = self.decoder.weight[:, self.neighbour_id]

        # b x n x N
        smoothed_basis = torch.empty((batch_size, self.n, self.N), device=self.device)
        for i in range(len(self.group_indices)):
            # b x n x N/m x mu
            bubbles = self.bubble(
                self.neighbour_distance[self.group_indices[i], :],
                bandwidths[:, :, self.group_indices[i]],
                self.mu,
            )
            # b x n x N/m
            smoothed_basis[:, :, self.group_indices[i]] = torch.sum(
                (
                    (basises[:, self.group_indices[i], :])
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1, -1)
                )
                * bubbles,
                dim=-1,
            )

        # batch size x 1 x n
        encoded = encoded.unsqueeze(2).permute((0, 2, 1))
        # batch size x N
        return torch.bmm(encoded, smoothed_basis).squeeze(1)

    def forward(self, x):
        return self.decode(self.encode(x))

    # distance: N/m x mu
    # w: b x n x N/m ([0 element_size])
    # mu: number of neighbour elements
    def bubble(self, distance, w, mu):
        b = w.shape[0]
        n = w.shape[1]
        window = torch.relu(
            -(distance.unsqueeze(0).unsqueeze(0).expand(b, n, -1, -1) ** 2)
            / (w.unsqueeze(-1) * mu) ** 2
            + 1
        )
        window = window / torch.sum(window, dim=-1, keepdim=True)
        # b x n x N/m x mu
        return window

    # def bubble(self, w):
    #     x = torch.arange(self.mu, device=self.device)
    #     window = torch.relu(-(x**2) / (w * self.mu) ** 2 + 1)
    #     window = window / torch.sum(window)
    #     return window

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
    def __init__(
        self,
        N,
        n,
        mu,
        m,
        neighbour_id,
        neighbour_distance,
        clustering_labels,
        group_indices,
        device,
    ):
        super(EncoderDecoder, self).__init__()
        self.nrbs = NRBS(
            N=N,
            n=n,
            mu=mu,
            m=m,
            neighbour_id=neighbour_id,
            neighbour_distance=neighbour_distance,
            clustering_labels=clustering_labels,
            group_indices=group_indices,
            device=device,
        ).to(device)
        self.device = device

    def train(self, train_data_loader, effective_batch=200, epochs=1):

        optim = torch.optim.Adam(self.nrbs.parameters(), 1e-5)
        loss_func = torch.nn.MSELoss(reduction="sum")
        best_loss = float("inf")
        self.nrbs.train()
        accu_itr = effective_batch // train_data_loader.batch_size

        for i in range(epochs):
            curr_loss = 0

            for j, x in enumerate(tqdm.tqdm(train_data_loader)):
                x = x.to(self.device)
                approximates = self.nrbs(x[:, : self.nrbs.N])
                loss = loss_func(x[:, self.nrbs.N :], approximates)
                loss.backward()
                if ((j + 1) % accu_itr == 0) or (j + 1 == len(train_data_loader)):
                    optim.step()
                    optim.zero_grad()
            with torch.no_grad():
                for x in tqdm.tqdm(train_data_loader):
                    x = x.to(self.device)
                    approximates = self.nrbs(x[:, : self.nrbs.N])
                    loss = loss_func(x[:, self.nrbs.N :], approximates)
                    curr_loss = curr_loss + loss.item()
            print("Itr {:}, loss = {:}".format(i, curr_loss / 1000))
            if curr_loss < best_loss:
                if os.path.isfile("models/nrbs_n_m_test.pth"):
                    os.remove("models/nrbs_n_m_test.pth")
                torch.save(self.nrbs, "models/nrbs_n_m_test.pth")

    def forward(self, x):
        return self.nrbs(x)
