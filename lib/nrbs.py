import torch
from functorch import vmap
import tqdm

torch.set_default_dtype(torch.float64)


class NRBS(torch.nn.Module):
    def __init__(self, N, n, mu, neighbours):
        super(NRBS, self).__init__()
        self.N = N
        self.n = n
        self.mu = mu
        self.neighbours = neighbours

        self.encoder = torch.nn.Linear(self.N, self.n)
        # self.decoder = torch.nn.Linear(self.n, self.N)
        self.decoder = torch.nn.Parameter(
            torch.Tensor(self.n, self.N).uniform_(-0.01, 0.01), requires_grad=True
        )
        self.bandwidth = torch.nn.Parameter(
            torch.Tensor(self.n, self.N).uniform_(-0.01, 0.01), requires_grad=True
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, encoded):
        vmap_bubble = vmap(self.bubble, in_dims=0)
        vmap_vmap_bubble = vmap(vmap_bubble, in_dims=0)
        # n x N x mu
        bubbles = vmap_vmap_bubble(self.bandwidth)
        # print("bubbles shape: ", bubbles.shape)
        smoothed_basis = self.smooth_basis(bubbles=bubbles)
        # return torch.matmul(encoded, self.decoder)
        return torch.matmul(encoded, smoothed_basis)

    def forward(self, x):
        return self.decode(self.encode(x))

    # def bubble(self, w):
    #     x = torch.arange(2 * self.mu)
    #     window = torch.relu(-((x - self.mu) ** 2) / (w * self.mu) ** 2 + 1)
    #     window = window / torch.sum(window)
    #     return window

    def bubble(self, w):
        x = torch.arange(self.mu)
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
            ]
        ).squeeze(2)

    def smooth_vec(self, basis_idx, node_idx, bubbles):
        return (
            self.decoder[basis_idx][self.getNeighbours(node_idx)]
            * bubbles[basis_idx][node_idx]
        ).sum(dim=1, keepdims=True)

    def smooth(self, basis_idx, node_idx, bubbles):
        return torch.sum(
            self.decoder[basis_idx][self.getNeighbours(node_idx)]
            * bubbles[basis_idx][node_idx]
        )


# class EncoderDecoder(torch.nn.Module):
#     def __init__(self, nrbs):
#         super(EncoderDecoder, self).__init__()
#         self.nrbs = nrbs

#     def train(self, Train_loader, epochs=1):

#         optim = torch.optim.Adam(self.nrbs.parameters(), 1e-5)
#         loss_function = torch.nn.BCEWithLogitsLoss()

#         best_acc = float("inf")
#         for i in range(epochs):
#             print("Epoch ", i)
#             self.nrbs.train()
#             curr_loss = 0
#             for j, X in enumerate(tqdm.notebook.tqdm(Train_loader)):
#                 input_id = X[0]
#                 mask = X[1]
#                 label = X[2]
#                 input_id = input_id.to(device)
#                 mask = mask.to(device)
#                 label = label.to(device)

#                 outputs = self.rs_model((input_id, mask))

#                 loss = loss_function(outputs, (label.unsqueeze(1)).float())
#                 loss.backward()
#                 curr_loss = curr_loss + loss

#                 if ((j + 1) % accu_itr == 0) or (j + 1 == len(TokenizedRSTrain_loader)):
#                     print("accumulated batch loss:", curr_loss)
#                     optim.step()
#                     optim.zero_grad()
#                     # lr_scheduler.step()
#                     curr_loss = 0

#             curr_acc = self.evaluate(TokenizedRSDev_loader)
#             torch.save(self.rs_model, "rs_fever_" + str(i) + ".pth")
