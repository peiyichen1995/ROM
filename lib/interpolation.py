import torch


def interpolate_values(phi, dof_values):
    return torch.matmul(phi, dof_values)


def interpolate_gradients(grad_phi, dof_values):
    return torch.einsum("...ijk,...j", grad_phi, dof_values)
