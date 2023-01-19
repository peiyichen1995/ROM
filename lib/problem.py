import torch

import qrule
import shape
import dofmap
import interpolation
import scipy
import numpy
import meshio

from functorch import vmap


class FEProblem:
    def __init__(self, mesh):
        self.mesh = mesh
        self.device = mesh.device
        qweights = torch.stack(
            [qrule.qweights(elem_type, 1, self.device) for elem_type in mesh.elem_type]
        )
        qpoints = torch.stack(
            [qrule.qpoints(elem_type, 1, self.device) for elem_type in mesh.elem_type]
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
        self.interpolated_values = torch.empty(
            self.mesh.num_elems(), device=self.device
        )
        self.interpolated_gradients = torch.empty(
            self.mesh.num_elems(), device=self.device
        )

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

    def register_coupled_value(self, weakform, variable_name):
        if variable_name not in weakform.coupled_variables:
            weakform.coupled_variables.append(variable_name)

        if variable_name not in self.coupled_values:
            self.coupled_values.append(variable_name)

        return self.coupled_values.index(variable_name)

    def register_coupled_gradient(self, weakform, variable_name):
        if variable_name not in weakform.coupled_variables:
            weakform.coupled_variables.append(variable_name)

        if variable_name not in self.coupled_gradients:
            self.coupled_gradients.append(variable_name)

        return self.coupled_gradients.index(variable_name)

    def residual(self):
        res = scipy.sparse.coo_array((self.dofmap.num_dofs(), 1), dtype=numpy.float64)
        col_idx = numpy.zeros_like(self.mesh.connectivity.cpu()).flatten()
        for variable in self.variables:
            row_idx = (
                self.dofmap.node_dof(variable, self.mesh.connectivity).cpu().flatten()
            )
            for weakform in self.weakforms:
                if weakform.variable is not variable:
                    continue
                res += scipy.sparse.coo_array(
                    (
                        vmap(weakform.evaluate)(
                            self.phis,
                            self.grad_phis,
                            self.JxWs,
                            self.interpolated_values,
                            self.interpolated_gradients,
                        )
                        .cpu()
                        .flatten(),
                        (
                            row_idx,
                            col_idx,
                        ),
                    ),
                    shape=(self.dofmap.num_dofs(), 1),
                )

        return res.todense()

    def jacobian(self):
        jac = scipy.sparse.coo_array(
            (self.dofmap.num_dofs(), self.dofmap.num_dofs()), dtype=numpy.float64
        )
        for ivar in self.variables:
            row_idx = numpy.repeat(
                self.dofmap.node_dof(ivar, self.mesh.connectivity).cpu(),
                self.mesh.connectivity.shape[1],
                axis=1,
            ).flatten()
            for jvar in self.variables:
                col_idx = numpy.tile(
                    self.dofmap.node_dof(jvar, self.mesh.connectivity).cpu(),
                    [1, self.mesh.connectivity.shape[1]],
                ).flatten()
                for weakform in self.weakforms:
                    if (
                        weakform.variable is not ivar
                        or jvar not in weakform.coupled_variables
                    ):
                        continue
                    jac += scipy.sparse.coo_array(
                        (
                            vmap(
                                lambda phi, grad_phi, JxW, value, gradient: weakform.d_evaluate(
                                    jvar, phi, grad_phi, JxW, value, gradient
                                )
                            )(
                                self.phis,
                                self.grad_phis,
                                self.JxWs,
                                self.interpolated_values,
                                self.interpolated_gradients,
                            )
                            .cpu()
                            .flatten(),
                            (
                                row_idx,
                                col_idx,
                            ),
                        ),
                        shape=(self.dofmap.num_dofs(), self.dofmap.num_dofs()),
                    )

        return jac.tocsc()

    def add_weakform(self, weakform):
        self.weakforms.append(weakform)

    def add_variable(self, variable_name, ic):
        self.dofmap.add_variable(variable_name)
        self.ics[variable_name] = ic
        self.variables.append(variable_name)

    def init_solution(self):
        self.sol = torch.empty(self.dofmap.num_dofs(), device=self.device)
        for variable_name, ic in self.ics.items():
            self.sol[self.dofmap.dofs(variable_name)] = vmap(ic)(self.mesh.nodes)
        self.interpolate()

    def solution(self):
        return self.sol

    def solution(self, variable_name):
        return self.sol[self.dofmap.dofs(variable_name)]

    def write_vtk(self, file_name):

        output = meshio.Mesh(
            self.mesh.nodes.cpu(),
            [("quad", self.mesh.connectivity.cpu())],
            point_data={
                variable: self.solution(variable).cpu() for variable in self.variables
            },
        )

        output.write(file_name)
