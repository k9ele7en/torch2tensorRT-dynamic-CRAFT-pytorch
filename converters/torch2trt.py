# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________

This file contain code for converting pretrain Pytorch model into TensorRT engine
_____________________________________________________________________________
"""
from icecream import ic
import sys
import os
from pathlib import Path

import torch
import numpy as np
import argparse

from torch2trt_dynamic import torch2trt_dynamic

from utils import experiment_loader, initial_logger, copyStateDict, get_cfg_defaults

sys.path.append("../")
from craft import CRAFT

logger = initial_logger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build(args):
    """Load the network and export it to tensorRT

    Parameters
    ----------
    args.weight : str
        Path to pretrain model of CRAFT (default is ./weights)
    args.dynamic : bool
        A flag used to apply dynamic input shape for TensorRT engine (default is
        True)

    Returns
    -------
    None
        Output is saved in ./weights/*.engine, no return
    """
    logger.info("Converting CRAFT Pytorch pth to TensorRT engine...")

    model_path, model_config = experiment_loader(model_format='pth', data_path=args.weight)
    
    # Load config come with pretrain model
    cfg_detec = get_cfg_defaults()
    cfg_detec.merge_from_file(model_config)
    cfg_detec.INFERENCE.TRT_DYNAMIC = args.dynamic
    # Set output path for tensorRT files
    output_path = Path('../weights/')

    # Set name for tensorRT files
    output_detec = os.path.join(output_path, "detec_rt.engine")
    
    # Dummy input data for models
    input_tensor = torch.randn((1, 3, 768, 768), requires_grad=False)
    input_tensor=input_tensor.cuda()
    input_tensor=input_tensor.to(device=device)

    # Load net
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(model_path)))
    net = net.cuda()
    net.eval()

    # Convert the model into tensorRT
    opt_shape_param = [
        [
            cfg_detec.INFERENCE.TRT_MIN_SHAPE,   # min
            cfg_detec.INFERENCE.TRT_OPT_SHAPE,   # opt
            cfg_detec.INFERENCE.TRT_MAX_SHAPE    # max
        ]
    ]
    
    if cfg_detec.INFERENCE.TRT_DYNAMIC:
        model_trt = torch2trt_dynamic(net, [input_tensor], fp16_mode=cfg_detec.INFERENCE.TRT_AMP, opt_shape_param=opt_shape_param)
    else:
        model_trt = torch2trt_dynamic(net, [input_tensor], fp16_mode=cfg_detec.INFERENCE.TRT_AMP)

    logger.info("Compare Pytorch output vs TensorRT engine output...")

    y = net(input_tensor)
    y_trt = model_trt(input_tensor)
    ic('Pytorch pth output: ', y)
    ic('TensorRT engine output: ', y_trt)
    with open(output_detec, "wb") as f:
        f.write(model_trt.engine.serialize())

    logger.info("Convert CRAFT Pytorch pth to TensorRT engine sucess")

def main():
    parser = argparse.ArgumentParser(description="Exports model to TensorRT, and post-processes it to insert TensorRT plugins")
    parser.add_argument("--weight", required=False, help="Path to input model folder", default='../weights')
    parser.add_argument("--dynamic", required=False, help="Use dynamic or not", default=True)
    
    args=parser.parse_args()

    build(args)

if __name__ == '__main__':
    main()