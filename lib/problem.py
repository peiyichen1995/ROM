import torch

import qrule
import shape
import dofmap
import interpolation
import meshio

from functorch import vmap


class FEProblem:
    def __init__(self, mesh):
        self.mesh = mesh
        qweights = torch.stack(
            [qrule.qweights(elem_type, 1) for elem_type in mesh.elem_type]
        )
        qpoints = torch.stack(
            [qrule.qpoints(elem_type, 1) for elem_type in mesh.elem_type]
        )
        self.phis = torch.stack(
            [
                shape.phi(elem_type, qpoint)
                for qpoint, elem_type in zip(qpoints, mesh.elem_type)
            ]
        )
        parametric_grad_phis = torch.stack(
            [
                shape.grad_phi(elem_type, qpoint)
                for qpoint, elem_type in zip(qpoints, mesh.elem_type)
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
                    mesh.elem_type, jacobians, qweights
                )
            ]
        )

        self.dofmap = dofmap.DofMap(mesh)
        self.ics = {}
        self.variables = []
        self.weakforms = []
        self.coupled_values = []
        self.coupled_gradients = []
        self.interpolated_values = torch.empty(self.mesh.num_elems())
        self.interpolated_gradients = torch.empty(self.mesh.num_elems())

    def interpolate(self):
        # num_elems x num_qp x num_coupled_values
        if self.coupled_values:
            self.interpolated_values = torch.stack(
                [
                    vmap(interpolation.interpolate_values)(
                        self.phis,
                        self.sol[
                            self.dofmap.node_dof(variable_name, self.mesh.connectivity)
                        ],
                    )
                    for variable_name in self.coupled_values
                ],
                dim=-1,
            )

        # num_elems x num_qp x 3 x num_coupled_gradients
        if self.coupled_gradients:
            self.interpolated_gradients = torch.stack(
                [
                    vmap(interpolation.interpolate_gradients)(
                        self.grad_phis,
                        self.sol[
                            self.dofmap.node_dof(variable_name, self.mesh.connectivity)
                        ],
                    )
                    for variable_name in self.coupled_gradients
                ],
                dim=-1,
            )

    def register_coupled_value(self, variable_name):
        self.coupled_values.append(variable_name)
        return len(self.coupled_values) - 1

    def register_coupled_gradient(self, variable_name):
        self.coupled_gradients.append(variable_name)
        return len(self.coupled_gradients) - 1

    def residual(self):
        for weakform in self.weakforms:
            return vmap(weakform.evaluate)(
                self.phis,
                self.grad_phis,
                self.JxWs,
                self.interpolated_values,
                self.interpolated_gradients,
            )
        # assemble residual using dofmap

    def add_weakform(self, weakform):
        self.weakforms.append(weakform)

    def add_variable(self, variable_name, ic):
        self.dofmap.add_variable(variable_name)
        self.ics[variable_name] = ic
        self.variables.append(variable_name)

    def init_solution(self):
        self.sol = torch.empty(self.dofmap.num_dofs())
        for variable_name, ic in self.ics.items():
            self.sol[self.dofmap.dofs(variable_name)] = vmap(ic)(self.mesh.nodes)
        self.interpolate()

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
