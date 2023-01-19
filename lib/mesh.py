import torch
from enum import IntEnum


class ElemType(IntEnum):
    Line2 = 0
    Quad4 = 1


class Mesh:
    def __init__(self, device):
        self.device = device

    def elems(self):
        return self.nodes[self.connectivity]

    def num_nodes(self):
        return len(self.nodes)

    def num_elems(self):
        return len(self.connectivity)


class LineMesh(Mesh):
    def __init__(self, params, device):
        super().__init__(device)
        self.nodes = torch.column_stack(
            (
                torch.linspace(
                    params["xmin"], params["xmax"], params["N"] + 1, device=self.device
                ),
                torch.zeros((params["N"] + 1, 1), device=self.device),
                torch.zeros((params["N"] + 1, 1), device=self.device),
            )
        )

        self.connectivity = torch.column_stack(
            (
                torch.arange(0, params["N"], device=self.device),
                torch.arange(1, params["N"] + 1, device=self.device),
            )
        )

        self.elem_type = torch.full(
            (params["N"], 1), ElemType.Line2, device=self.device
        )


class RectangleMesh(Mesh):
    def __init__(self, params, device):
        super().__init__(device)
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(
                params["xmin"], params["xmax"], params["Nx"] + 1, device=self.device
            ),
            torch.linspace(
                params["ymin"], params["ymax"], params["Ny"] + 1, device=self.device
            ),
            indexing="ij",
        )
        self.nodes = torch.column_stack(
            (grid_x.flatten(), grid_y.flatten(), torch.zeros_like(grid_x.flatten()))
        )

        self.connectivity = torch.empty(
            (params["Nx"] * params["Ny"], 4), dtype=torch.int64, device=self.device
        )
        for i in range(params["Nx"]):
            for j in range(params["Ny"]):
                elem_id = i * params["Ny"] + j
                self.connectivity[elem_id, 2] = elem_id + i
                self.connectivity[elem_id, 1] = elem_id + i + 1
                self.connectivity[elem_id, 3] = elem_id + i + 1 + params["Ny"]
                self.connectivity[elem_id, 0] = elem_id + i + 2 + params["Ny"]

        self.elem_type = torch.full(
            (params["Nx"] * params["Ny"], 1), ElemType.Quad4, device=self.device
        )
