import torch
from mesh import ElemType


def phi(elem_type, x):
    if elem_type == ElemType.Line2:
        return torch.stack(((1 - x[..., 0]) / 2, (x[..., 0] + 1) / 2), dim=-1)
    if elem_type == ElemType.Quad4:
        return torch.stack(
            (
                0.25 * (1 + x[..., 0]) * (1 + x[..., 1]),
                0.25 * (1 - x[..., 0]) * (1 + x[..., 1]),
                0.25 * (1 - x[..., 0]) * (1 - x[..., 1]),
                0.25 * (1 + x[..., 0]) * (1 - x[..., 1]),
            ),
            dim=-1,
        )


def grad_phi(elem_type, x):
    batch_size = len(x)
    if elem_type == ElemType.Line2:
        return torch.tensor([[[-1.0 / 2, 0, 0], [1.0 / 2, 0, 0]]]).expand(
            batch_size, -1, -1
        )
    if elem_type == ElemType.Quad4:
        return torch.stack(
            (
                torch.stack(
                    (
                        0.25 * (1 + x[..., 1]),
                        0.25 * (1 + x[..., 0]),
                        torch.zeros((batch_size)),
                    ),
                    dim=-1,
                ),
                torch.stack(
                    (
                        -0.25 * (1 + x[..., 1]),
                        0.25 * (1 - x[..., 0]),
                        torch.zeros((batch_size)),
                    ),
                    dim=-1,
                ),
                torch.stack(
                    (
                        -0.25 * (1 - x[..., 1]),
                        -0.25 * (1 - x[..., 0]),
                        torch.zeros((batch_size)),
                    ),
                    dim=-1,
                ),
                torch.stack(
                    (
                        0.25 * (1 - x[..., 1]),
                        -0.25 * (1 + x[..., 0]),
                        torch.zeros((batch_size)),
                    ),
                    dim=-1,
                ),
            ),
            dim=1,
        )


# elem: num_nodes x 3
# grad: num_qp x num_nodes x 3
# return: num_qp x 3 x 3
def jacobian(elem, grad):
    return torch.matmul(elem.T.unsqueeze(0), grad)
