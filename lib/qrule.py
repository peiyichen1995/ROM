import torch
from mesh import ElemType
import math


def qpoints(elem_type, order):
    if elem_type == ElemType.Line2 and order == 1:
        return torch.tensor([[-1 / math.sqrt(3), 0, 0], [1 / math.sqrt(3), 0, 0]])
    if elem_type == ElemType.Quad4 and order == 1:
        return torch.tensor(
            [
                [-1 / math.sqrt(3), -1 / math.sqrt(3), 0],
                [1 / math.sqrt(3), -1 / math.sqrt(3), 0],
                [-1 / math.sqrt(3), 1 / math.sqrt(3), 0],
                [1 / math.sqrt(3), 1 / math.sqrt(3), 0],
            ]
        )


def qweights(elem_type, order):
    if elem_type == ElemType.Line2 and order == 1:
        return torch.ones(2)
    if elem_type == ElemType.Quad4 and order == 1:
        return torch.ones(4)


def JxW(elem_type, jacobian, qweight):
    svdvals = torch.linalg.svdvals(jacobian)
    if elem_type == ElemType.Line2:
        return svdvals[..., 0] * qweight
    if elem_type == ElemType.Quad4:
        return svdvals[..., 0] * svdvals[..., 1] * qweight
