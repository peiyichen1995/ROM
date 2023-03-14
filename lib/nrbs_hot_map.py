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

        self.encoder1 = torch.nn.Linear(self.N, 100)
        self.encoder2 = torch.nn.Linear(100, self.n)

        self.decoder = torch.nn.Linear(self.n, self.N)
        hotness_map = []
        in_dim = self.n
        for dim in self.m:
            hotness_map.append(torch.nn.Linear(in_dim, dim))
            in_dim = dim
        self.hotness_map = torch.nn.ModuleList(hotness_map)

        self.B = torch.nn.Parameter(torch.tensor([B0]))

        torch.nn.init.kaiming_normal_(self.encoder1.weight, mode="fan_out")
        torch.nn.init.kaiming_normal_(self.encoder2.weight, mode="fan_out")
        torch.nn.init.kaiming_normal_(self.decoder.weight, mode="fan_out")

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

    def eval(self, data_loader, norm):
        loss_func = torch.nn.MSELoss(reduction="mean")
        proj_loss_func = torch.nn.MSELoss(reduction="sum")

        mean_err = 0
        proj_err = 0
        max_difference = 0
        with torch.no_grad():
            for u in data_loader:
                approximates = self.nrbs(u)

                loss = loss_func(u, approximates)
                mean_err = mean_err + loss.item()

                proj_loss = proj_loss_func(u, approximates)
                proj_err = proj_err + proj_loss.item()

                max_difference = max(
                    torch.max(torch.abs(u - approximates)), max_difference
                )
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        return (
            mean_err / len(data_loader),
            np.sqrt(proj_err) / torch.sqrt(norm),
            max_difference,
        )

    def train(
        self,
        train_data_loader,
        test_data_loader,
        train_norm,
        eval_norm,
        comment,
        model_name,
        effective_batch=240,
        epochs=1,
    ):
        writer = SummaryWriter(comment=comment)
        loss_func = torch.nn.MSELoss(reduction="mean")

        lr = 1e-3
        optim = torch.optim.Adam(self.nrbs.parameters(), lr=lr)

        accu_itr = effective_batch // train_data_loader.batch_size

        (
            best_train_mean_err,
            best_train_proj_err,
            best_train_max_abs_difference,
        ) = self.eval(train_data_loader, train_norm)
        (
            best_eval_mean_err,
            best_eval_proj_err,
            best_eval_max_abs_difference,
        ) = self.eval(test_data_loader, eval_norm)

        writer.add_scalar("loss/train_current", best_train_mean_err, -1)
        writer.add_scalar("loss/test_current", best_eval_mean_err, -1)
        writer.add_scalar(
            "projection_err/train_current",
            best_train_proj_err,
            -1,
        )
        writer.add_scalar(
            "projection_err/test_current",
            best_eval_proj_err,
            -1,
        )
        writer.add_scalar(
            "max_abs_difference/train_current", best_train_max_abs_difference, -1
        )
        writer.add_scalar(
            "max_abs_difference/test_current", best_eval_max_abs_difference, -1
        )

        patience = 0
        for i in range(epochs):
            for j, u in enumerate(train_data_loader):

                approximates = self.nrbs(u)
                objective = loss_func(u, approximates)
                objective.backward()

                if ((j + 1) % accu_itr == 0) or (j + 1 == len(train_data_loader)):
                    optim.step()
                    optim.zero_grad()
                torch.cuda.empty_cache()

            torch.cuda.empty_cache()

            train_mean_err, train_proj_err, train_max_abs_difference = self.eval(
                train_data_loader, train_norm
            )
            eval_mean_err, eval_proj_err, eval_max_abs_difference = self.eval(
                test_data_loader, eval_norm
            )

            if eval_proj_err < best_eval_proj_err:
                patience = 0
                best_eval_proj_err = eval_proj_err
                if os.path.isfile(model_name):
                    os.remove(model_name)
                torch.save(self.nrbs, model_name)
            else:
                patience = patience + 1
            if patience == 10:
                patience = 0
                lr = lr / 10
                optim = torch.optim.Adam(self.nrbs.parameters(), lr=lr)
            writer.add_scalar("loss/train_current", train_mean_err, i)
            writer.add_scalar("loss/test_current", eval_mean_err, i)
            writer.add_scalar(
                "projection_err/train_current",
                train_proj_err,
                i,
            )
            writer.add_scalar(
                "projection_err/test_current",
                eval_proj_err,
                i,
            )
            writer.add_scalar(
                "projection_err/test_best",
                best_eval_proj_err,
                i,
            )
            writer.add_scalar(
                "max_abs_difference/train_current", train_max_abs_difference, i
            )
            writer.add_scalar(
                "max_abs_difference/test_current", eval_max_abs_difference, i
            )
            writer.add_scalar("lr", lr, i)

    def forward(self, x):
        return self.nrbs(x)
