import torch
from monai.losses import FocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, target, smooth =1 ):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = target.view(-1)
        intersection = (inputs*targets).sum()
        dice = (2.0 * intersection + smooth)/(inputs.sum()+targets.sum()+smooth)
        print(dice)
        return torch.clamp((1-dice),0,1)


class BCEFocalLosswithLogits(nn.Module):
      def __init__(self,gamma =0.2, alpha =0.6 , reduction ='mean'):
            super(BCEFocalLosswithLogits, self).__init__()
            self.gamma  = gamma
            self.alpha = alpha
            self.reduction = reduction

      def forward(self, input, label):
            #input: [N,H,W], label:[N,H,W]
            inputs = F.sigmoid(input)
            label[label>0] = 1.0
            print("inputs_shape: ",inputs.shape)
            print("label size: ", label.shape)
            print("1-inputs_size: " ,(1-inputs).shape)
            print("1-inputs*label_shape: ",((1-inputs)*label).shape)

            alpha = self.alpha
            gamma = self.gamma
            loss = -alpha*(1-inputs)**gamma*label*torch.log(inputs) - (1-alpha)*inputs**gamma*(1-label)*torch.log(1-inputs) + 1e-6
            print("loss_shape: ", loss.shape)
            if self.reduction == 'mean':
                  loss = loss.mean()
            elif self.reduction =='sum':
                  loss = loss.sum()
            print(loss)
            return loss 

class TverskyLoss(nn.Module):    
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        smooth = 1e-5
        dice = 0.
        for i in range(pred.size(1)):
            dice += (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / ((pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1)+
                        0.3 * (pred[:,i] * (1 - target[:,i])).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * ((1 - pred[:,i]) * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2)


class CrossEntropyFocalLoss(nn.Module):
      def __init__(self, alpha = None, gamma = 0.2, reduction ='mean'):
            super(CrossEntropyFocalLoss, self).__init__()
            self.reduction = reduction
            self.alpha = alpha
            self.gamma = gamma

      def forward(self, inputs, label):
            #inputs: [N,C,D,H,W], label: [N,D,H,W]
            #loss = sum(-y_i * log(C_i))
            if inputs.dim()>2:
                  inputs = inputs.view(inputs.size(0),inputs.size(1),-1) #[N,C,DHW]
                  inputs = inputs.transpose(1,2) #[N,DHW,C]
                  inputs = inputs.contiguous().view(-1, inputs.size(2)) #[NDHW, C]
            label = label.view(-1,1) #[NDHW, 1]
            label = label.type(torch.int64)

            pt = F.softmax(inputs, dim =1)
            pt = pt.gather(1, label).view(-1) #[NDHW]
            log_pt = torch.log(pt)

            if self.alpha is not None:
                  alpha = self.alpha.gather(0,label.view(-1)) #[NDHW]
                  log_pt = log_pt * alpha

            loss = -1 *(1-pt) ** self.gamma * log_pt

            if self.reduction == 'mean':
                  loss =loss.mean()

            elif self.reduction =='sum':
                  loss = loss.sum()
            return loss


class CrossEntropyLoss(torch.nn.Module):
      def __init__(self, reduction = 'mean'):
            super(CrossEntropyLoss, self).__init__()
            self.reduction = reduction
      def forward(self, inputs, label):
            #inputs: [N, C, D, H, W], label:[N, D, H, W]
            #loss = sum(-y_i * log(C_i))
            if inputs.dim()>2:
                  inputs = inputs.view(inputs.size(0),inputs.size(1),-1) #[N, C, DHW]
                  inputs = inputs.transpose(1, 2) #[N, DHW, C]
                  inputs = inputs.contiguous().view(-1, inputs.size(2)) #[NDHW, C]

            label = label.view(-1, 1) #[NDHW, 1]

            inputs = F.log_softmax(inputs, 1)
            inputs = inputs.gather(1, label) #[NDHW, 1]
            loss = -1*inputs

            if self.reduction == 'mean':
                  loss = loss.mean()
            elif self.reduction == 'sum':
                  loss = loss.sum()
            print("lose_shape: ",loss.shape)

            return loss