import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
import copy

import sys

sys.path.append("lib")

import lib

import torch

torch.set_default_dtype(torch.float64)

from functorch import vmap
from torch.utils.data import DataLoader
import tqdm

device = torch.device("cuda")

coords_x, coords_y, _ = lib.utils.read_coords("2d_burger_data/time_step_0.csv")
coords = torch.stack((coords_x, coords_y), dim=1)

dt = 0.002
num_steps = 1001
num_nodes = coords_x.shape[0]

datas = lib.utils.read_data(num_steps, num_nodes, "2d_burger_data/time_step_", "vel_0")
datas.shape

u_dot = lib.utils.u_dot(datas, dt)

datas = torch.hstack((datas[:-1], u_dot))

m = 50
clustering = KMeans(n_clusters=m, random_state=0, n_init="auto").fit(coords)

N = datas.shape[1] // 2
n = 20
# fixed support (length)
mu = int(np.ceil(N / 100))
neighbour_distance, neighbour_id = lib.utils.topk_neighbours(coords, mu)

batch_size = 6
datas = datas.to(device)
train_data = DataLoader(datas, batch_size=batch_size, shuffle=True)

ed = lib.nrbs_n_m.EncoderDecoder(
    N=N,
    n=n,
    mu=mu,
    m=m,
    neighbour_id=neighbour_id,
    neighbour_distance=neighbour_distance,
    clustering_labels=torch.tensor(clustering.labels_).type(torch.LongTensor),
    device=device,
)

ed.train(train_data_loader=train_data, epochs=1000)
