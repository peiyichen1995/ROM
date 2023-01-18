import torch


class DofMap:
    def __init__(self, mesh):
        self.mesh = mesh
        self.variable_id = {}

    def add_variable(self, variable_name):
        self.variable_id[variable_name] = len(self.variable_id)

    def node_dof(self, variable_name, node_id):
        return self.variable_id[variable_name] * self.mesh.num_nodes() + node_id

    def elem_dofs(self, variable_name, elem_id):
        return self.node_dof(variable_name, self.mesh.connectivity[elem_id])

    def dofs(self, variable_name):
        return self.node_dof(variable_name, torch.arange(0, self.mesh.num_nodes()))

    def num_dofs(self):
        return len(self.variable_id) * self.mesh.num_nodes()
