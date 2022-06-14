import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_loss(task_loss_type):
    if task_loss_type == "cross_entropy_loss":
        criterion = cross_entropy_loss
    elif task_loss_type == "nll_loss":
        criterion = nn.NLLLoss()
    elif task_loss_type == "focal_loss":
        criterion = FocalLoss()
    elif task_loss_type == "multi_label_cross_entropy_loss":
        criterion = multi_label_cross_entropy_loss
    elif task_loss_type == "mse_loss":
        criterion = mse_loss
    elif task_loss_type == "mae_loss":
        criterion = mae_loss
    elif task_loss_type == "bce_loss":
        criterion = bce_loss
    else:
        raise NotImplementedError

    return criterion


def bce_loss(outputs, labels):
    return nn.BCELoss(reduction='mean')(outputs, labels)

def mse_loss(outputs, labels):
    return nn.MSELoss(reduction='mean')(outputs, labels)

def mae_loss(outputs, labels):
    return nn.L1Loss(reduction='mean')(outputs, labels)

def multi_label_cross_entropy_loss(outputs, labels):
    labels = labels.float()
    temp = outputs
    res = - labels * torch.log(temp) - (1 - labels) * torch.log(1 - temp)
    res = torch.mean(torch.sum(res, dim=1))

    return res


def cross_entropy_loss(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, labels)



    
def binary_cross_entropy_with_logits(input, target):
        """Sigmoid Activation + Binary Cross Entropy

        @param input: torch.Tensor (size N)
        @param target: torch.Tensor (size N)
        @return loss: torch.Tensor (size N)
        """
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(
                target.size(), input.size()))

        return (torch.clamp(input, 0) - input * target 
                + torch.log(1 + torch.exp(-torch.abs(input))))
    
def cross_entropy(input, target, eps=1e-6):
        """k-Class Cross Entropy (Log Softmax + Log Loss)
        
        @param input: torch.Tensor (size N x K)
        @param target: torch.Tensor (size N x K)
        @param eps: error to add (default: 1e-6)
        @return loss: torch.Tensor (size N)
        """
        if not (target.size(0) == input.size(0)):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(0), input.size(0)))

        log_input = F.log_softmax(input + eps, dim=1)
        y_onehot = Variable(log_input.data.new(log_input.size()).zero_())
        y_onehot = y_onehot.scatter(1, target.unsqueeze(1), 1)
        loss = y_onehot * log_input
        return -loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        # if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        # if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
