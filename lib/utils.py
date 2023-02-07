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


def read_coords(file_name):
    file_name = file_name
    data = pd.read_csv(file_name)
    coords_x = data["mesh_model_coordinates_0"].to_numpy()
    coords_y = data["mesh_model_coordinates_1"].to_numpy()
    coords_z = data["mesh_model_coordinates_2"].to_numpy()
    return coords_x, coords_y, coords_z
