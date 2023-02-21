import torch
from functorch import vmap
import tqdm
import numpy as np
import os

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
        self.test_decoder = torch.nn.Linear(self.n, self.N)
        self.decoder = torch.nn.Parameter(torch.Tensor(self.n, self.N))
        self.bandwidth_layers = torch.nn.Linear(self.n, self.n * self.m)

    def encode(self, x):
        return self.encoder(x)

    # # distance: N x mu
    # # w: b x n x N ([0 element_size])
    # # mu: number of neighbour elements
    # def bubble(self, neighbour_distance, w, mu):
    #     b = w.shape[0]
    #     window = torch.relu(
    #         -(
    #             neighbour_distance.unsqueeze(0).unsqueeze(0).expand(b, self.n, -1, -1)
    #             ** 2
    #         )
    #         / (w.unsqueeze(-1) * mu) ** 2
    #         + 1
    #     )
    #     window = window / torch.sum(window, dim=1, keepdim=True)
    #     # n x N x mu
    #     return window

    # # x: n x N
    # # neighbour_id: N x mu
    # # w: b x n x N
    # def convolve(self, x, neighbour_id, neighbour_distance, w, mu):
    #     b = w.shape[0]
    #     # b x n x N x mu
    #     bubbles = self.bubble(neighbour_distance, w, mu)
    #     return torch.sum(
    #         x[:, neighbour_id].unsqueeze(0).expand(b, -1, -1, -1) * bubbles, dim=-1
    #     )

    # def decode(self, encoded):
    #     # b x n x m
    #     bandwidths = self.bandwidth_layers(encoded)
    #     bandwidths = bandwidths.reshape(-1, self.n, self.m)

    #     # b x n x N
    #     bandwidths = bandwidths[:, :, self.clustering_labels]

    #     # b x n x N
    #     convolved_basis = self.convolve(
    #         self.decoder,
    #         self.neighbour_id,
    #         self.neighbour_distance,
    #         bandwidths,
    #         self.mu,
    #     )
    #     # batch size x N
    #     return torch.bmm(encoded.unsqueeze(1), convolved_basis).squeeze(1)

    # distance: N x mu
    # w: n x N ([0 element_size])
    # mu: number of neighbour elements
    def bubble(self, neighbour_distance, w, mu):
        window = torch.relu(
            -(neighbour_distance.unsqueeze(0).expand(self.n, -1, -1) ** 2)
            / (w.unsqueeze(-1) * mu) ** 2
            + 1
        )
        window = window / torch.sum(window, dim=1, keepdim=True)
        # n x N x mu
        return window

    # x: n x N
    # neighbour_id: N x mu
    # w: n x N
    def convolve(self, x, neighbour_id, neighbour_distance, w, mu):
        # n x N x mu
        bubbles = self.bubble(neighbour_distance, w, mu)
        return torch.sum(x[:, neighbour_id] * bubbles, dim=-1)

    def decode(self, encoded):
        # return self.test_decoder(encoded)
        # n x m
        bandwidths = torch.sigmoid(self.bandwidth_layers(encoded)) / 60
        bandwidths = bandwidths.reshape(self.n, self.m)

        # n x N
        bandwidths = bandwidths[:, self.clustering_labels]

        # # n x N
        # convolved_basis = self.convolve(
        #     self.decoder,
        #     self.neighbour_id,
        #     self.neighbour_distance,
        #     bandwidths,
        #     self.mu,
        # )

        # n x N x mu
        bubbles = self.bubble(self.neighbour_distance, bandwidths, self.mu)

        convolved_basis = torch.stack(
            [
                torch.sum(self.decoder[i][self.neighbour_id] * bubbles[i], dim=-1)
                for i in range(self.n)
            ],
            dim=0,
        )

        # batch size x N
        return torch.matmul(encoded, convolved_basis)

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

    def train(self, train_data_loader, effective_batch=50, epochs=1):

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
            history_size=4,
            max_iter=4,
            line_search_fn="strong_wolfe",
        )

        # optim = torch.optim.Adam(self.nrbs.parameters(), 1e-2)

        accu_itr = effective_batch // train_data_loader.batch_size

        for i in range(epochs):
            self.nrbs.train()
            curr_loss = 0
            for j, x in enumerate(tqdm.tqdm(train_data_loader)):
                x = x.to(self.device)
                approximates = self.nrbs(x)
                lbfgs.zero_grad()
                objective = torch.sum(loss_func(x, approximates))
                objective.backward(retain_graph=True)
                lbfgs.step(closure)

                # if ((j + 1) % accu_itr == 0) or (j + 1 == len(train_data_loader)):
                #     optim.step()
                #     optim.zero_grad()
                # # optim.step()
                # # optim.zero_grad()

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
