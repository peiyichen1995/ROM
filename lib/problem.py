import torch

import qrule
import shape
import dofmap
import meshio

from functorch import vmap


class FEProblem:
    def __init__(self, mesh):
        self.mesh = mesh
        self.qweights = torch.stack(
            [qrule.qweights(elem_type, 1) for elem_type in mesh.elem_type]
        )
        self.qpoints = torch.stack(
            [qrule.qpoints(elem_type, 1) for elem_type in mesh.elem_type]
        )
        self.phis = torch.stack(
            [
                shape.phi(elem_type, qpoint)
                for qpoint, elem_type in zip(self.qpoints, mesh.elem_type)
            ]
        )
        parametric_grad_phis = torch.stack(
            [
                shape.grad_phi(elem_type, qpoint)
                for qpoint, elem_type in zip(self.qpoints, mesh.elem_type)
            ]
        )
        jacobians = torch.stack(
            [
                shape.jacobian(elem, grad_phi)
                for elem, grad_phi in zip(mesh.elems(), parametric_grad_phis)
            ]
        )
        self.grad_phis = torch.matmul(
            parametric_grad_phis, torch.linalg.pinv(jacobians)
        )
        self.JxWs = torch.stack(
            [
                qrule.JxW(elem_type, jacobian, qweight)
                for elem_type, jacobian, qweight in zip(
                    mesh.elem_type, jacobians, self.qweights
                )
            ]
        )

        self.dofmap = dofmap.DofMap(mesh)
        self.ics = {}
        self.variables = []

    def add_variable(self, variable_name, ic):
        self.dofmap.add_variable(variable_name)
        self.ics[variable_name] = ic
        self.variables.append(variable_name)

    def init_solution(self):
        self.sol = torch.empty(self.dofmap.num_dofs())
        for variable_name, ic in self.ics.items():
            self.sol[self.dofmap.dofs(variable_name)] = vmap(ic)(self.mesh.nodes)

    def solution(self):
        return self.sol

    def solution(self, variable_name):
        return self.sol[self.dofmap.dofs(variable_name)]

    def write_vtk(self, file_name):

        output = meshio.Mesh(
            self.mesh.nodes,
            [("quad", self.mesh.connectivity)],
            point_data={
                variable: self.solution(variable) for variable in self.variables
            },
        )

        output.write(file_name)
