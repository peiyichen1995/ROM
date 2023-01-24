import torch

import qrule
import shape
import dofmap
import interpolation
import bc
import scipy
import numpy
import meshio
from pathlib import Path

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
        self.coupled_values_old = []
        self.coupled_gradients = []
        self.interpolated_values = torch.empty(
            self.mesh.num_elems(), device=self.device
        )
        self.interpolated_values_old = torch.empty(
            self.mesh.num_elems(), device=self.device
        )
        self.interpolated_gradients = torch.empty(
            self.mesh.num_elems(), device=self.device
        )
        self.dt = torch.zeros(self.mesh.num_elems(), device=self.device)
        self.nodal_bcs = []
        self.time = torch.zeros(self.mesh.num_elems(), device=self.device)

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

        # num_elems x num_qp x num_coupled_values
        if self.coupled_values_old:
            self.interpolated_values_old = torch.stack(
                [
                    vmap(interpolation.interpolate_values)(
                        self.phis,
                        self.sol_old[
                            self.dofmap.node_dof(variable_name, self.mesh.connectivity)
                        ],
                    )
                    for variable_name in self.coupled_values_old
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

    def register_coupled_value_old(self, weakform, variable_name):
        if variable_name not in self.coupled_values_old:
            self.coupled_values_old.append(variable_name)
        return self.coupled_values_old.index(variable_name)

    def register_coupled_gradient(self, weakform, variable_name):
        if variable_name not in weakform.coupled_variables:
            weakform.coupled_variables.append(variable_name)

        if variable_name not in self.coupled_gradients:
            self.coupled_gradients.append(variable_name)

        return self.coupled_gradients.index(variable_name)

    def residual(self):
        res = torch.sparse_coo_tensor(
            torch.empty([1, 0]), [], [self.dofmap.num_dofs()], device=self.device
        )
        for variable in self.variables:
            row_idx = (
                self.dofmap.node_dof(variable, self.mesh.connectivity)
                .flatten()
                .unsqueeze(0)
            )
            for weakform in self.weakforms:
                if weakform.variable is not variable:
                    continue
                res += torch.sparse_coo_tensor(
                    row_idx,
                    vmap(weakform.evaluate)(
                        self.phis,
                        self.grad_phis,
                        self.JxWs,
                        self.interpolated_values,
                        self.interpolated_gradients,
                        self.interpolated_values_old,
                        self.dt,
                    ).flatten(),
                    [self.dofmap.num_dofs()],
                )

        res = res.to_dense()
        for bc in self.nodal_bcs:
            bc.apply_residual(res)

        return res.cpu()

    def jacobian(self):
        jac = torch.sparse_coo_tensor(
            torch.empty([2, 0]),
            [],
            [self.dofmap.num_dofs(), self.dofmap.num_dofs()],
            device=self.device,
        )
        for ivar in self.variables:
            row_idx = torch.repeat_interleave(
                self.dofmap.node_dof(ivar, self.mesh.connectivity),
                self.mesh.connectivity.shape[1],
                dim=1,
            ).flatten()
            for jvar in self.variables:
                col_idx = (
                    self.dofmap.node_dof(jvar, self.mesh.connectivity)
                    .repeat([1, self.mesh.connectivity.shape[1]])
                    .flatten()
                )
                for weakform in self.weakforms:
                    if (
                        weakform.variable is not ivar
                        or jvar not in weakform.coupled_variables
                    ):
                        continue

                    jac += torch.sparse_coo_tensor(
                        torch.stack([row_idx, col_idx]),
                        vmap(weakform.d_evaluate)(
                            torch.full(
                                (self.mesh.num_elems(),),
                                self.variables.index(jvar),
                                device=self.device,
                            ),
                            self.phis,
                            self.grad_phis,
                            self.JxWs,
                            self.interpolated_values,
                            self.interpolated_gradients,
                            self.interpolated_values_old,
                            self.dt,
                        ).flatten(),
                        [self.dofmap.num_dofs(), self.dofmap.num_dofs()],
                    )
        jac = jac.coalesce()
        jac = scipy.sparse.coo_array(
            (jac.values().cpu(), jac.indices().cpu()), shape=jac.shape
        ).tocsr()
        for bc in self.nodal_bcs:
            bc.apply_jacobian(jac)

        return jac

    def add_weakform(self, weakform):
        self.weakforms.append(weakform)

    def add_variable(self, variable_name, ic):
        self.dofmap.add_variable(variable_name)
        self.ics[variable_name] = ic
        self.variables.append(variable_name)

    def init_solution(self):
        self.sol = torch.empty(self.dofmap.num_dofs(), device=self.device)
        for variable_name, ic in self.ics.items():
            self.sol[self.dofmap.dofs(variable_name)] = vmap(
                ic, randomness="different"
            )(self.mesh.nodes).reshape(-1)
        self.sol_old = self.sol.clone()
        self.interpolate()

    def solution(self):
        return self.sol

    def solution(self, variable_name):
        return self.sol[self.dofmap.dofs(variable_name)]

    def add_nodal_bc(self, bc):
        self.nodal_bcs.append(bc)

    def solution_add(self, x):
        self.sol += torch.tensor(x, device=self.device)
        self.interpolate()

    def advance_in_time(self, dt):
        self.sol_old = self.sol.clone()
        self.interpolate()
        self.time += dt
        self.dt = torch.full((self.mesh.num_elems(),), dt, device=self.device)

    def solve(self, atol=1e-8, rtol=1e-6, max_itr=20):
        i = 0
        r = self.residual()
        r_norm = torch.linalg.norm(r)
        r_norm_0 = r_norm.item()
        print("Iteration {:}, ||r|| = {:.3E}".format(i, r_norm.item()))
        converged = False
        while not converged:
            i += 1
            if i > max_itr:
                raise Exception("Maximum iteration reached.")
            J = self.jacobian()
            self.solution_add(-scipy.sparse.linalg.spsolve(J, r))
            r = self.residual()
            r_norm = torch.linalg.norm(r)
            print("Iteration {:}, ||r|| = {:.3E}".format(i, r_norm.item()))
            converged = r_norm.item() < atol or r_norm.item() < r_norm_0 * rtol

    def print_time(self):
        print(
            "time = {:.6E}, dt = {:.6E}".format(
                self.time.cpu()[0].item(), self.dt.cpu()[0].item()
            )
        )

    def write_step(self, file_name, step):
        file = Path(file_name)
        file = file.stem + "_{:}".format(step) + file.suffix

        output = meshio.Mesh(
            self.mesh.nodes.cpu(),
            [("quad", self.mesh.connectivity.cpu())],
            point_data={
                variable: self.solution(variable).cpu() for variable in self.variables
            },
        )

        output.write(file)


# burgers
# nm rom

# complex problem
