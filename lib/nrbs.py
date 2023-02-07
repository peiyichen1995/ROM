import torch


class NRBS(torch.nn.Module):
    def __init__(self, N, n, encoder_dims, decoder_dims):
        super(NRBS, self).__init__()
        self.encoder = []
        in_dim = N
        for encoder_dim in encoder_dims:
            self.encoder.append(torch.nn.Linear(in_dim, encoder_dim))
            in_dim = encoder_dim
        self.encoder.append(torch.nn.Linear(in_dim, n))

        self.decoder = []
        in_dim = n
        for decoder_dim in decoder_dims:
            self.decoder.append(torch.nn.Linear(in_dim, decoder_dim))
            in_dim = decoder_dim
        self.decoder.append(torch.nn.Linear(in_dim, N))

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))
