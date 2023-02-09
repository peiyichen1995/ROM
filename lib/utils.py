import torch
import pandas as pd

# number of data points x size of one data sample
def read_data(num_steps, num_nodes, file_base, var_name, file_extension=".csv"):
    datas = torch.zeros((num_steps, num_nodes))
    for i in range(num_steps):
        file_name = file_base + str(i) + file_extension
        data = pd.read_csv(file_name)
        data = data[var_name].to_numpy()
        datas[i, :] = torch.tensor(data)

    return datas


def u_dot(datas, dt):
    u_n_plus_1 = datas[1:]
    u_n = datas[:-1]
    u_dot = (u_n_plus_1 - u_n) / dt
    return u_dot


def read_coords(file_name):
    file_name = file_name
    data = pd.read_csv(file_name)
    coords_x = torch.tensor(data["mesh_model_coordinates_0"].to_numpy())
    coords_y = torch.tensor(data["mesh_model_coordinates_1"].to_numpy())
    coords_z = torch.tensor(data["mesh_model_coordinates_2"].to_numpy())
    return coords_x, coords_y, coords_z


# (x - y)^2 = x^2 - 2*x*y + y^2
def similarity_matrix(mat):
    # get the product x * y
    # here, y = x.t()
    r = torch.mm(mat, mat.t())
    # get the diagonal elements
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    # compute the distance matrix
    D = diag + diag.t() - 2 * r
    return D.sqrt()


def topk_neighbours(coords, k):
    sim = similarity_matrix(coords)
    indices = torch.topk(sim, k, largest=False, sorted=True)[1]
    return indices
