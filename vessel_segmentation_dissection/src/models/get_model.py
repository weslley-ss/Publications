import sys
import torch
from .res_unet_adrian import UNet as unet
from .dic_unet import dic_unet

import json 
def dic_kernel(confs, in_c=3):
    
    kernels, dilations, residuals, downsamplings = eval(confs)
    
    paddings = [k//2 for k in kernels]

    params = {  
        "FR": {
            "in_channels": in_c,
            "out_channels": 8,
            "kernel_size": residuals[0],
            "padding": 0,
            "dilation": 1,
            "bias": False
        },
        "FC": [
            {
                "in_channels": in_c,
                "out_channels": 8,
                "kernel_size": kernels[0],
                "padding": "same",
                "dilation": dilations[0]
            },
            {
                "in_channels": 8,
                "out_channels": 8,
                "kernel_size": kernels[1],
                "padding": "same",
                "dilation": dilations[1]
            }
        ],
        "D1": {
            "kernel_size": downsamplings[0],
            "stride": downsamplings[0]
        },
        "B1R": {
            "in_channels": 8,
            "out_channels": 16,
            "kernel_size": residuals[1],
            "padding": 0,
            "dilation": 1,
            "bias": False
        },
        "B1C": [
            {
                "in_channels": 8,
                "out_channels": 16,
                "kernel_size": kernels[2],
                "padding": "same",
                "dilation": dilations[2]
            },
            {
                "in_channels": 16,
                "out_channels": 16,
                "kernel_size": kernels[3],
                "padding": "same",
                "dilation": dilations[3]
            }
        ],
        "D2": {
            "kernel_size": downsamplings[1],
            "stride": downsamplings[1]
        },
        "B2R": {
            "in_channels": 16,
            "out_channels": 32,
            "kernel_size": residuals[2],
            "padding": 0,
            "dilation": 1,
            "bias": False
        },
        "B2C": [
            {
                "in_channels": 16,
                "out_channels": 32,
                "kernel_size": kernels[4],
                "padding": "same",
                "dilation": dilations[4]
            },
            {
                "in_channels": 32,
                "out_channels": 32,
                "kernel_size": kernels[5],
                "padding": "same",
                "dilation": dilations[5]
            }
        ],
        "UP1": {
            "in_channels": 32,
            "out_channels": 16,
            "kernel_size": downsamplings[1],
            "stride": downsamplings[1]
        },
        "CB1": {
            "in_channels": 16,
            "out_channels": 16,
            "kernel_size": 3,
            "padding": 1,
            "dilation": 1
        },
        "UB1R": {
            "in_channels": 32,
            "out_channels": 16,
            "kernel_size": residuals[3],
            "padding": 0,
            "dilation": 1,
            "bias": False
        },
        "UB1C": [
            {
                "in_channels": 32,
                "out_channels": 16,
                "kernel_size": kernels[6],
                "padding": "same",
                "dilation": dilations[6]
            },
            {
                "in_channels": 16,
                "out_channels": 16,
                "kernel_size": kernels[7],
                "padding": "same",
                "dilation": dilations[7]
            }
        ],
        "UP2": {
            "in_channels": 16,
            "out_channels": 8,
            "kernel_size": downsamplings[0],
            "stride": downsamplings[0]
        },
        "CB2": {
            "in_channels": 8,
            "out_channels": 8,
            "kernel_size": 3,
            "padding": 1,
            "dilation": 1
        },
        "UB2R": {
            "in_channels": 16,
            "out_channels": 8,
            "kernel_size": residuals[4],
            "padding": 0,
            "dilation": 1,
            "bias": False
        },
        "UB2C": [
            {
                "in_channels": 16,
                "out_channels": 8,
                "kernel_size": kernels[8],
                "padding": "same",
                "dilation": dilations[8]
            },
            {
                "in_channels": 8,
                "out_channels": 8,
                "kernel_size": kernels[9],
                "padding": "same",
                "dilation": dilations[9]
            }
        ],
        "F": {
            "in_channels": 8,
            "out_channels": 1,
            "kernel_size": 1,
            "padding": 0,
            "dilation": 1
        }
    }
    
    return params

class wnet(torch.nn.Module):
    def __init__(self, n_classes=1, in_c=3, layers=(8,16,32), conv_bridge=True, shortcut=True, mode='train'):
        super(wnet, self).__init__()
        self.unet1 = unet(in_c=in_c, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
        self.unet2 = unet(in_c=in_c+n_classes, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
        self.n_classes = n_classes
        self.mode=mode

    def forward(self, x):
        x1 = self.unet1(x)
        x2 = self.unet2(torch.cat([x, x1], dim=1))
        if self.mode!='train':
            return x2
        return x1,x2

def get_arch(model_name, kernels, in_c=3, n_classes=1):

    if model_name == 'unet':
        model = unet(in_c=in_c, n_classes=n_classes, layers=[8,16,32], conv_bridge=True, shortcut=True)
    elif model_name == 'dic_unet':
        dic = dic_kernel(kernels, in_c)
        model = dic_unet(parametros = dic)
    elif model_name == 'big_unet':
        model = unet(in_c=in_c, n_classes=n_classes, layers=[12,24,48], conv_bridge=True, shortcut=True)
    elif model_name == 'wnet':
        model = wnet(in_c=in_c, n_classes=n_classes, layers=[8,16,32], conv_bridge=True, shortcut=True)
    elif model_name == 'big_wnet':
        model = wnet(in_c=in_c, n_classes=n_classes, layers=[8,16,32,64], conv_bridge=True, shortcut=True)
    else: sys.exit('not a valid model_name, check models/get_model.py')
    
    return model