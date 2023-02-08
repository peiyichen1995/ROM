import torch
from functorch import vmap
import tqdm

torch.set_default_dtype(torch.float64)


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
        # self.bandwidth = torch.nn.Parameter(
        #     torch.Tensor(self.n, self.N).uniform_(-0.01, 0.01), requires_grad=True
        # )
        self.bandwidth_layer = torch.nn.Linear(self.n, self.N, device=self.device)

    def encode(self, x):
        return self.encoder(x)

    # width not related to u hat
    # def decode_not_of_u(self, encoded):
    #     vmap_bubble = vmap(self.bubble, in_dims=0)
    #     vmap_vmap_bubble = vmap(vmap_bubble, in_dims=0)
    #     # n x N x mu
    #     bubbles = vmap_vmap_bubble(self.bandwidth)
    #     print("bubbles shape: ", bubbles.shape)
    #     smoothed_basis = self.smooth_basis(bubbles=bubbles)
    #     # return torch.matmul(encoded, self.decoder)
    #     return torch.matmul(encoded, smoothed_basis)

    def decode(self, encoded):
        vmap_bubble = vmap(self.bubble, in_dims=0)
        vmap_vmap_bubble = vmap(vmap_bubble, in_dims=0)
        # batch size x N x mu
        bubbles = vmap_vmap_bubble(self.bandwidth_layer(encoded))
        print("bubbles shape: ", bubbles.shape)
        # batch size x n x N
        smoothed_basis = self.smooth_basis(bubbles=bubbles)
        # batch size x 1 x n
        encoded = encoded.unsqueeze(2).permute((0, 2, 1))
        # batch size x N
        return torch.bmm(encoded, smoothed_basis).squeeze(1)

    def forward(self, x):
        return self.decode(self.encode(x))

    # def bubble(self, w):
    #     x = torch.arange(2 * self.mu)
    #     window = torch.relu(-((x - self.mu) ** 2) / (w * self.mu) ** 2 + 1)
    #     window = window / torch.sum(window)
    #     return window

    def bubble(self, w):
        x = torch.arange(self.mu, device=self.device)
        window = torch.relu(-(x**2) / (w * self.mu) ** 2 + 1)
        window = window / torch.sum(window)
        return window

    def getNeighbours(self, idx):
        return self.neighbours[idx]

    def convolve(self, x, bubble):
        return x * bubble

    # def smooth_basis_not_of_u(self, bubbles):
    #     node_idxs = torch.linspace(0, self.N - 1, self.N, dtype=torch.long)
    #     return torch.stack(
    #         [
    #             self.smooth_vec(i, node_idx=node_idxs, bubbles=bubbles)
    #             for i in range(self.n)
    #         ]
    #     ).squeeze(2)

    # def smooth_vec_not_of_u(self, basis_idx, node_idx, bubbles):
    #     return (
    #         self.decoder[basis_idx][self.getNeighbours(node_idx)]
    #         * bubbles[basis_idx][node_idx]
    #     ).sum(dim=1, keepdims=True)

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
    def __init__(self, N, n, mu, neighbours, device):
        super(EncoderDecoder, self).__init__()
        self.nrbs = NRBS(N=N, n=n, mu=mu, neighbours=neighbours, device=device).to(
            device
        )
        self.device = device

    def train(self, train_data_loader, epochs=1):

        optim = torch.optim.Adam(self.nrbs.parameters(), 1e-3)
        loss_func = torch.nn.MSELoss(reduction="none")

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

            torch.save(self.nrbs, "models/nrbs_" + str(i) + ".pth")
            print("Itr {:}, loss = {:}".format(i, curr_loss))

    def forward(self, x):
        return self.nrbs(x)
