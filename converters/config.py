# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________

This file configs to convert multiple model format for Triton
_____________________________________________________________________________
"""
from yacs.config import CfgNode as CN

_C = CN()

##################
#################
_C._BASE_ = CN()

###################
###################
# ---------------------------------------------------------------------------- #
# INFERENCE
# ---------------------------------------------------------------------------- #
_C.INFERENCE = CN()

  # Use dynamic input
_C.INFERENCE.TRT_DYNAMIC = True

_C.INFERENCE.TRT_MIN_SHAPE = [1,3,256,256]

_C.INFERENCE.TRT_OPT_SHAPE = [1,3,700,700]

_C.INFERENCE.TRT_MAX_SHAPE = [1,3,1200,1200]

  # Use mix-precision (FP16)
_C.INFERENCE.TRT_AMP = True

  # Workspace size for export engine process (in GB), default=5 GB
_C.INFERENCE.TRT_WORKSPACE = 5