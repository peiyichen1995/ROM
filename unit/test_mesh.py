import unittest
import sys

sys.path.append("lib")

from mesh import *


class TestMesh(unittest.TestCase):
    def test_LineMesh(self):
        mesh = LineMesh({"xmin": 0, "xmax": 1, "N": 3})

        self.assertTrue(
            torch.allclose(
                mesh.coordinates,
                torch.tensor([[0, 0, 0], [1.0 / 3, 0, 0], [2.0 / 3, 0, 0], [1, 0, 0]]),
            )
        )

        self.assertTrue(
            torch.equal(mesh.connectivity, torch.tensor([[0, 1], [1, 2], [2, 3]]))
        )


if __name__ == "__main__":
    unittest.main()
