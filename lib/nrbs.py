import torch
from functorch import vmap

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
        bubbles = vmap_vmap_bubble(self.bandwidth)
        print("bubbles shape: ", bubbles.shape)
        return torch.matmul(encoded, self.decoder)

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
