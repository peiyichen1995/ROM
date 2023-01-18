import problem
import torch


class WeakForm:
    pass


class Reaction(WeakForm):
    def __init__(self, problem, params):
        super().__init__()
        self.u = problem.register_coupled_value(params["variable"])

    def evaluate(self, phi, grad_phi, JxW, value, gradient):
        return torch.einsum("ij,i,i", phi, value[:, self.u], JxW)


class Diffusion(WeakForm):
    def __init__(self, problem, params):
        super().__init__()
        self.grad_u = problem.register_coupled_gradient(params["variable"])

    def evaluate(self, phi, grad_phi, JxW, value, gradient):
        return torch.einsum("ijk,ik,i", grad_phi, gradient[..., self.grad_u], JxW)
