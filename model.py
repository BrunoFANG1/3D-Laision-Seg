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
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
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
import argparse
from numpy.core.arrayprint import printoptions
import pynvml
import sys
from tqdm import tqdm

def get_args( args : list) -> dict:

    parser = argparse.ArgumentParser(description ='3dUnet command line argument parser')
    parser.add_argument('--mode',
                        help = 'the action you want to do [train],[test]',
                        type = str,
                        choices =["train", "predict"],
                        required = True)

    parser.add_argument('--model_save_name',
                        help = 'the training result model name',
                        type = str)

    parser.add_argument('--train_data_dir',
                        help = 'directory contains training data',
                        type = str)
    parser.add_argument('--label_data_dir',
                        help = 'directory contains label data',
                        type = str)

    parser.add_argument('--model_save_dir',
                        help = 'directory to save the model checkpoint',
                        type = str)
    parser.add_argument('--lr',
                        help = 'learning rate',
                        type = float)
    parser.add_argument('--loss_function',
                        help ='the loss function you want use',
                        type = str,
                        choices =["BCELoss", "CrossEntropy","BCEFocal","CrossEntropyFocal","DiceLoss"],
                        required  = True)
    parser.add_argument('--epochs',
                        help = 'Epochs for training',
                        type = int)
    parser.add_argument('--bs',
                        help = 'batch size',
                        type = int)
    parser.add_argument('--tensorboard_save_dir', 
                        help = 'The directory to save the tensorbroad data',
                        type = str)
    parser.add_argument('--Resume',
                        help = 'The training resume or not',
                        choices =["True", "False"],
                        type = str)
    parser.add_argument('--model_path',
                        help = 'the input training model save path',
                        type = str)
    options = vars(parser.parse_args())
    return options

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size =3, stride =1, padding =1),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size =3, stride =1, padding =1),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size =3, stride =1, padding =1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
            )
        self.relu = nn.ReLU(inplace = True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        #print(g1.shape)
        #print(x1.shape)
        g1 = F.interpolate(g1, scale_factor = (x1.shape[2]/g1.shape[2],x1.shape[3]/g1.shape[3],x1.shape[4]/g1.shape[4]), mode = 'trilinear')
        #print(g1.shape)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        #print(psi.shape)

        return x*psi

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(ch_in, ch_out, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x
    
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor =2),
            nn.Conv3d(ch_in,ch_out,3,stride =1 ,padding =1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inpalce =True)
        )
    def forward(self, x):
        x =self.up(x)
        return x    



class Att_Unet(nn.Module):
    def __init__(self, in_channel = 1, out_channel = 1, training = True):
        super(Att_Unet, self).__init__()
        self.trianing = training
        # encoder section
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channel, 16, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True),
            nn.Conv3d(16, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True)
        ).to("cuda:0")
        self.encoder2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True)
        ).to("cuda:0")

        self.encoder3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True),
            nn.Conv3d(64, 128, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace = True)
        ).to("cuda:0")

        self.encoder4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace = True)
        ).to("cuda:0")

        self.att1 = Attention_block(F_g =128, F_l =128, F_int =128).to("cuda:0")

        self.encoder5 = nn.Sequential(
            nn.Conv3d(128, 256, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace = True)
        ).to("cuda:0")

        # decoder section
        self.decoder1 = nn.Sequential(
            nn.Conv3d(384, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True)
        ).to('cuda:0')
        self.att2 = Attention_block(F_g=64, F_l=64, F_int =64).to("cuda:0")
        self.decoder2 = nn.Sequential(
            nn.Conv3d(64, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True)
        ).to('cuda:0')
        self.decoder3 = nn.Sequential(
            nn.Conv3d(128, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True)
        ).to('cuda:1')
        self.att3 = Attention_block(F_g =32, F_l=32, F_int =32).to("cuda:1")
        self.decoder4 = nn.Sequential(
            nn.Conv3d(32, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True)
        ).to('cuda:1')
        self.decoder5 = nn.Sequential(
            nn.Conv3d(64, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True)
        ).to('cuda:1')
        self.decoder6 = nn.Sequential(
            nn.Conv3d(32, 2, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace = True)
        ).to('cuda:1')
        self.decoder7 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1)
        ).to('cuda:1')


        
    def forward(self, x):
        # encoder section
        x = x.to("cuda:0")
        print('origin:',x.shape)
        x = self.encoder1(x).to("cuda:0")# relu(4->8)
        f1 = x #(8)
        f1= f1.to("cuda:1")
        print('f1:',x.shape)
        #print("encoder1_size:",f1.shape)
        x = F.max_pool3d(x,kernel_size = 2,stride = 2,padding = 0).to("cuda:0")# maxpool(8->8)
        #print("test",x.shape)
        x = self.encoder2(x).to("cuda:0")# relu(8->16)
        f2 = x #(16)
        f2 = f2.to("cuda:1")
        print('f2:',x.shape)
        #print("encoder2_size:",f2.shape)
        x = F.max_pool3d(x,kernel_size = 2,stride = 2,padding = 0).to("cuda:0")# maxpool(16->16)
        
        x = self.encoder3(x).to("cuda:0")# relu(16->32)
        f3 = x #(32)
        f3 =f3.to("cuda:0")
        print('f3:',x.shape)
        #print("encoder3_size:",f3.shape)
        x = F.max_pool3d(x,kernel_size = 2,stride = 2,padding = 0).to("cuda:0")# maxpool(32->32)
        x = self.encoder4(x).to("cuda:0")# relu(32->32)
        a1 = x
        x=x.to("cuda:0")
        a1=a1.to("cuda:0")
        print('x:',x.shape)
        A1 = self.att1(g = f3, x =a1).to("cuda:0")#attention block1
        print('A1:',A1.shape)
        x = self.encoder5(x).to("cuda:0")#relu(32->64)
        print('x:',x.shape)
        
        # decoder section

        x = F.interpolate(x, scale_factor = (f3.shape[2]/x.shape[2],f3.shape[3]/x.shape[3],f3.shape[4]/x.shape[4]), mode = 'trilinear').to("cuda:0")# upsample（16->16)
        print('x:',x.shape)
        A1 = F.interpolate(A1, scale_factor = (f3.shape[2]/A1.shape[2],f3.shape[3]/A1.shape[3],f3.shape[4]/A1.shape[4]), mode = 'trilinear').to("cuda:0")# upsample（16->16)
        print('A1:',A1.shape)
        x = torch.cat((x,A1),1).to("cuda:0") #(64+32 = 96)
        print('x:',x.shape)
        x = x.to("cuda:0")
        x = self.decoder1(x).to("cuda:0") # relu(96 ->32)
        print('x:',x.shape)

        a2 = x #attention block2, size = 32
        a2 = a2.to("cuda:0")
        print('a2:',a2.shape)
        print('f2:',f2.shape)
        f2 = f2.to('cuda:0')
        A2 = self.att2(g =f2,x=a2).to("cuda:0") # size =32
        print('A2',A2.shape)
        A2 = F.interpolate(A2, scale_factor = (f2.shape[2]/A2.shape[2],f2.shape[3]/A2.shape[3],f2.shape[4]/A2.shape[4]), mode = 'trilinear').to("cuda:1") # upsample（16->16)
      #   x = x.to("cuda:1")
        #print("decoder1_size:",x.shape)
        # x = x.to("cuda:1")
        x = self.decoder2(x).to("cuda:0") #relu(32->32)
        print('x:',x.shape)
        x= x.to("cuda:1")
        x = F.interpolate(x, scale_factor =(f2.shape[2]/x.shape[2],f2.shape[3]/x.shape[3],f2.shape[4]/x.shape[4]), mode = 'trilinear').to("cuda:1") # upsample(256->256)
        print('A2:',A2.shape)
        x = torch.cat((x,A2),1).to("cuda:1") #(4+4 = 8)
        # x = x.to("cuda:1")
        print('x:',x.shape)
        x = self.decoder3(x).to("cuda:1") # relu(8 ->2)
        print('x:',x.shape)
        #print("decoder2_size:",x.shape)
        a3 = x#attention block2
        print('a3:',a3.shape)
        f1 = f1.to("cuda:1")
        a3  = a3.to("cuda:1")
        A3 = self.att3(g =f1,x=a3).to("cuda:1")
        print('x:',x.shape)
        A3 = F.interpolate(A3, scale_factor = (f1.shape[2]/A3.shape[2],f1.shape[3]/A3.shape[3],f1.shape[4]/A3.shape[4]), mode = 'trilinear').to("cuda:1") # upsample（16->16)
        print('A3:',A3.shape)
        # x = x.to("cuda:1")
        x = self.decoder4(x).to("cuda:1") #relu(2->2)
        print('x:',x.shape)
        x = F.interpolate(x, scale_factor =(f1.shape[2]/x.shape[2],f1.shape[2]/x.shape[2],f1.shape[2]/x.shape[2]), mode = 'trilinear').to("cuda:1") # upsample(128->128)
        print('x:',x.shape)
        x = torch.cat((x,A3),1).to("cuda:1") #(2+2 = 4)
        print('x:',x.shape)
        # x= x.to("cuda:2")
        x = self.decoder5(x).to("cuda:1") # relu(4 ->2)
        print('x:',x.shape)
        #print("decoder3_size:",x.shape)
        x = self.decoder6(x).to("cuda:1")
        print('x:',x.shape)
        x = self.decoder7(x).to("cuda:1")
        print('x:',x.shape)
        x = F.interpolate(x, scale_factor = (f1.shape[2]/x.shape[2],f1.shape[3]/x.shape[3],f1.shape[4]/x.shape[4]), mode = 'trilinear').to("cuda:1")


        return x   



# In[ ]:
lossBCE = nn.CrossEntropyLoss()

class BCEDiceLoss(nn.Module):
    def __init__(self, a, b):
        super(BCEDiceLoss, self).__init__()
        self.a = a
        self.bce = nn.BCEWithLogitsLoss()
        self.b = b
        self.dice = DiceLoss()
    def forward(self, input, target):
        inputs = F.sigmoid(input)
        inputs = inputs.view(-1)
        targets = target.view(-1)  
        return self.a*self.bce(inputs,targets)+self.b*self.dice(inputs, targets)
        





class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, target, smooth =1 ):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = target.view(-1)
        intersection = (inputs*targets).sum()
        dice = (2.0 * intersection + smooth)/(inputs.sum()+targets.sum()+smooth)
        return torch.clamp((1-dice),0,1)

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
 

        
        


      

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path,patience=15, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'earlystopping_model.pt')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
        
        


from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)
def train(model, model_name, loss_metric, lr, epochs, train_dataloader, val_dataloader, test_dataloader, tensor_save,**kwargs):
    net = model
    if torch.cuda.device_count()>1:
        print('Lets use',torch.cuda.device_count(),'GPU!')
    writer = SummaryWriter(tensor_save)
    optimizer = torch.optim.AdamW(net.parameters(), lr = lr,weight_decay = 1e-5)
    #loss_metric = loss_fcn
    val_loss_old = 10000
    val_loss_unchanged_counter = 0
    #Common.print_network(net)# keep modify
    train_loss_list = []
    train_dice_list =[]
    val_loss_list = []
    val_dice_list = []
    train_name_list =[]
    val_name_list =[]
    test_name_list =[]
    # step_sche = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)
    print("Start Training")
    print(print_network(net))

    for epoch in tqdm(range(start_epoch+1,epochs), total = epochs):
        print(f"==========Epoch:{epoch}==========lr:{lr}========{epoch}/{epochs}")
        loss_sum = 0
        dice_sum = 0
        train_batch_count = 0
        val_batch_count = 0
        if epoch == 50:
            lr = 0.00005
        if epoch == 100:
            lr = 0.00003
        if epoch == 150:
            lr = 0.00001

             
        for index, (ctid, mriid, img, mask, mask_onehot)  in enumerate(train_dataloader):
            img, mask = img.float(), mask.float()
            img, mask = img, mask.to("cuda:1")
            net.train()
            optimizer.zero_grad()
            # optimizer.step()
            # step_sche.step()
            output = net.forward(img)
            print("output shape: ",output.shape)
            print("mask shape: ",mask.shape)
            loss = loss_fcn(output, mask)
            # if loss.data() == nan:
            #       break
            dice = 1-loss
            loss.backward()
            print("train_loss: ", loss)
            loss_sum += loss
            dice_sum += dice
            optimizer.step()
            train_name_list.append(ctid)
            train_batch_count = train_batch_count+1
        train_loss = loss_sum.item()/train_batch_count
        train_dice = dice_sum.item()/train_batch_count
        writer.add_scalar("train loss",float(train_loss),epoch)
        train_loss_list.append(train_loss)
        train_dice_list.append(train_dice)
            
        with torch.no_grad():
            loss_sum = 0
            dice_sum = 0
            for index,(ctid, mriid, img, mask, mask_onehot) in enumerate(val_dataloader):
                img, mask = img.float(), mask.float()
                img, mask = img, mask.to("cuda:1")
                net.eval()
                output = net.forward(img)
                loss = loss_fcn(output, mask)
                print("val_loss: ", loss)
                dice = 1-loss
                loss_sum += loss
                dice_sum += dice 
                val_name_list.append(ctid)
                val_batch_count = val_batch_count+1
            print("val_batch_count: ",val_batch_count)
            val_loss = loss_sum.item()/val_batch_count
            val_dice = dice_sum.item()/val_batch_count
            val_loss_list.append(val_loss)
            writer.add_scalar("valid loss",float(val_loss),epoch)
            writer.add_scalar("valid dice",float(val_dice),epoch)
            early_stopping(val_loss,net)
            if early_stopping.early_stop:
                print("Early Stopping")
                break
            val_dice_list.append(val_dice)
            # writer.add_scalars("train loss/valid loss",{"train loss": float(train_loss),"valid loss": float(val_loss)},epoch)

            for index,(ctid, mriid, img, mask, mask_onehot) in enumerate(test_dataloader):
                test_name_list.append(ctid)



            print('Epoch:',epoch+1)
            print('-'*20)
            print('Train Loss:',train_loss)
            print('Train Dice Score:',train_dice)
            print('-'*20)
            print('Validation Loss:',val_loss)
            print('Validation Dice Score:',val_dice)
            print('-'*20)
            torch.save({
            'epoch': epoch+1,
            'model': model,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss_list,
            'val_dice': val_dice_list,
            'train_dice': train_dice_list,
            'train_loss': train_loss_list,
            'train_name': train_name_list,
            'val_name':val_name_list,
            'test_name': test_name_list
            },os.path.join(MODEL_SAVE_DIR,f"{BESTMODEL_NAME}.pt"))
            torch.save(net.state_dict(),os.path.join(MODEL_SAVE_DIR,f"{BESTMODEL_NAME}_state.pt"))

            print("model saved")

# if __name__ == '__main__':
arguments = get_args(sys.argv)
MODE = arguments.get('mode')
TRAIN_DIR = arguments.get('train_data_dir')
LABEL_DIR = arguments.get('label_data_dir')
LOSS_FCN = arguments.get('loss_function')
TENSOR = arguments.get('tensorboard_save_dir')
if MODE == 'train':
    BESTMODEL_NAME = arguments.get('model_save_name')
    MODEL_SAVE_DIR = arguments.get('model_save_dir')
    LEARNING_RATE = arguments.get('lr')
    EPOCHS = arguments.get('epochs')
    BATCH_SIZE = arguments.get('bs')
    RESUME_TIP = arguments.get('Resume')
    MODEL_PATH  = arguments.get('model_path')
    early_stopping = EarlyStopping(MODEL_SAVE_DIR)

    CT_path = TRAIN_DIR
    MRI_path = LABEL_DIR
    train_set = CTDataset(CT_image_root = CT_path, MRI_label_root = MRI_path)
    train_set_size = int(train_set.__len__()*0.8)
    test_set_size = len(train_set) - train_set_size
    train_set, test_set = data.random_split(train_set, [train_set_size, test_set_size])
    train_set_size1 = int(len(train_set)*0.8)
    valid_set_size = train_set_size - train_set_size1
    train_set, valid_set = data.random_split(train_set, [train_set_size1, valid_set_size])
    print("Train data set:",len(train_set))
    print("Valid data set:", len(valid_set))
    print("Test data set:", len(test_set))
    


    Batch_Size = BATCH_SIZE
    train_loader = DataLoader(dataset = train_set, batch_size = Batch_Size,shuffle=True)
    valid_loader = DataLoader(dataset = valid_set, batch_size = Batch_Size,shuffle=True)
    test_loader = DataLoader(dataset = test_set, batch_size = Batch_Size)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
    if RESUME_TIP == 'True':
      path_checkpoint = MODEL_PATH
      checkpoint = torch.load(path_checkpoint)
      model = Att_Unet()
      model.load_state_dict(checkpoint)
      optimizer = optim.Adam
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
    elif RESUME_TIP == 'False':
         model = Att_Unet()
         start_epoch = 0
         



    

    if LOSS_FCN  == 'BCEFocalLoss':
          pass

    model_name = "Attunet_NewData"
    loss_fcn = FocalLoss(to_onehot_y = False)

    output = train(model, 
                   model_name, 
                   loss_metric = loss_fcn, 
                   lr =LEARNING_RATE, 
                   epochs = EPOCHS, 
                   train_dataloader = train_loader,
                   val_dataloader = valid_loader,
                   test_dataloader = test_loader,
                   tensor_save = TENSOR)

print('Train Finished')


