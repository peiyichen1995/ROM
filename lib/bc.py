class BC:
    def __init__(self, params):
        self.variable = params["variable"]
        self.coupled_variables = []


class DirichletBC(BC):
    def __init__(self, problem, params):
        super().__init__(params)
        self.problem = problem
        self.value = params["value"]
        self.boundary = params["boundary"]

    def apply_residual(self, residual):
        dofs = self.problem.dofmap.node_dof(self.variable, self.boundary)
        residual[dofs] = self.problem.sol[dofs] - self.value

    def apply_jacobian(self, jacobian):
        dofs = self.problem.dofmap.node_dof(self.variable, self.boundary.cpu())
        for dof in dofs:
            jacobian.data[jacobian.indptr[dof] : jacobian.indptr[dof + 1]] = 0
            jacobian[dof, dof] = 1
        jacobian.eliminate_zeros()
