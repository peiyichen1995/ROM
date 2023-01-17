import torch

import qrule
import shape
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
        self.variables = {}

    def add_variable(self, variable_name, ic):
        self.variables[variable_name] = vmap(ic)(self.mesh.coordinates)


# dofmap
