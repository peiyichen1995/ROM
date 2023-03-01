import mesh
import qrule
import shape
import problem
import dofmap
import weakform
import interpolation
import bc
import torch
import nrbs
import nrbs_n_N
import nrbs_n_1
import nrbs_n_m
import utils

import nrbs_test
import nrbs_no_convolve
import nrbs_N

torch.set_default_dtype(torch.float64)
