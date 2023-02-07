import torch

torch.set_default_dtype(torch.float64)


class NRBS(torch.nn.Module):
    def __init__(self, N, n):
        super(NRBS, self).__init__()
        self.N = N
        self.n = n

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
        return torch.matmul(encoded, self.decoder)

    def forward(self, x):
        return self.decode(self.encode(x))
