#!/usr/bin/env python
# coding: utf-8

# In[ ]:
"""
Author: Joe
Date: 08-10-2022
Edited: 12-06-2022

Attention Unet parallel training on multiple GPU with larger model
*08-11: Add the BalanceDataParallel to balance the dataparallel problem
*12-06: Add the onehot code on this versioin
*01-30: Arrange onehot code (Attentional Unet) and use the different loss function
*02-20: Add model parallel function to the code trainining
*03-08: Increase model to add more parameters: 3M-5M
*04-12: Write a code to resume the training after time pause
"""
from cmath import nan
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms

#from Dataset import CTDataset
import torch
from tqdm import tqdm
#from Unet_3d_model import Unet_3d
import numpy as np
import SimpleITK as sitk
import torchvision 
import torch.utils.data as data
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from numpy.core.arrayprint import printoptions
import pynvml
import sys
from tqdm import tqdm

# In[ ]:
lossBCE = nn.CrossEntropyLoss()
        

"""
 Following Loss Function is the source code from
 https://github.com/wolny/pytorch-3dunet
 """       
def flatten(tensor):
      C = tensor.size(1)
      axis_order = (1, 0) + tuple(range(2, tensor.dim()))
      transposed = tensor.permute(axis_order)
      return transposed.contiguous().view(C,-1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import numpy as np
import torch

        
        



