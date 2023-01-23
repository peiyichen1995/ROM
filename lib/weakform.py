import problem
import torch


class WeakForm:
    def __init__(self, params):
        self.variable = params["variable"]
        self.coupled_variables = []


class Reaction(WeakForm):
    def __init__(self, problem, params):
        super().__init__(params)
        self.u = problem.register_coupled_value(self, params["variable"])

    # element residual
    def evaluate(self, phi, grad_phi, JxW, value, gradient):
        return torch.einsum("ij,i,i", phi, value[:, self.u], JxW)

    def d_evaluate(self, jvar, phi, grad_phi, JxW, value, gradient):
        return torch.einsum("ij,ik,i->jk", phi, phi, JxW)


class Diffusion(WeakForm):
    def __init__(self, problem, params):
        super().__init__(params)
        self.grad_u = problem.register_coupled_gradient(self, params["variable"])

    def evaluate(self, phi, grad_phi, JxW, value, gradient):
        return torch.einsum("ijk,ik,i", grad_phi, gradient[..., self.grad_u], JxW)

    def d_evaluate(self, jvar, phi, grad_phi, JxW, value, gradient):
        return torch.einsum("ijk,ilk,i->jl", grad_phi, grad_phi, JxW)


# time derivative
