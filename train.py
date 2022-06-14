# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
from folder import ImageFolder
import torch.backends.cudnn as cudnn
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
# from PIL import Image
import copy
import time
import os
from model import two_view_net, three_view_net
from random_erasing import RandomErasing
from autoaugment import ImageNetPolicy, CIFAR10Policy
import yaml
from shutil import copyfile
from utils import update_average, get_model_list, load_network, save_network, make_weights_for_balanced_classes
from pytorch_metric_learning import losses, miners  # pip install pytorch-metric-learning
from circle_loss import CircleLoss, convert_label_to_similarity

version = torch.__version__
# fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError:  # will be 3.x series
    print(
        'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='two_view', type=str, help='output model name')
parser.add_argument('--pool', default='avg', type=str, help='pool avg')
parser.add_argument('--data_dir', default='./data/train', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--h', default=384, type=int, help='height')
parser.add_argument('--w', default=384, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='the number of views')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_NAS', action='store_true', help='use NAS')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation')
parser.add_argument('--resume', action='store_true', help='use resume trainning')
parser.add_argument('--share', action='store_true', help='share weight between different view')
parser.add_argument('--extra_Google', action='store_true', help='using extra noise Google')
parser.add_argument('--fp16', action='store_true',
                    help='use float16 instead of float32, which will save about 50% memory')
parser.add_argument('--arcface', action='store_true', help='use ArcFace loss')
parser.add_argument('--circle', action='store_true', help='use Circle loss')
parser.add_argument('--cosface', action='store_true', help='use CosFace loss')
parser.add_argument('--contrast', action='store_true', help='use contrast loss')
parser.add_argument('--triplet', action='store_true', help='use triplet loss')
parser.add_argument('--lifted', action='store_true', help='use lifted loss')
parser.add_argument('--sphere', action='store_true', help='use sphere loss')
parser.add_argument('--loss_merge', action='store_true', help='combine perspectives to calculate losses')
opt = parser.parse_args()

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0

fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#

transform_train_list = [
    # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_satellite_list = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomAffine(90),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(opt.h, opt.w), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list
    transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                       hue=0)] + transform_satellite_list

if opt.DA:
    transform_train_list = [ImageNetPolicy()] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
    'satellite': transforms.Compose(transform_satellite_list)}

train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = {}
image_datasets['satellite'] = datasets.ImageFolder(os.path.join(data_dir, 'satellite'),
                                                   data_transforms['satellite'])
image_datasets['street'] = datasets.ImageFolder(os.path.join(data_dir, 'street'),
                                                data_transforms['train'])
image_datasets['drone'] = datasets.ImageFolder(os.path.join(data_dir, 'drone'),
                                               data_transforms['train'])
image_datasets['google'] = ImageFolder(os.path.join(data_dir, 'google'),
                                       # google contain empty subfolder, so we overwrite the Folder
                                       data_transforms['train'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=2, pin_memory=True)  # 8 workers may work faster
               for x in ['satellite', 'street', 'drone', 'google']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'street', 'drone', 'google']}
class_names = image_datasets['street'].classes
print(dataset_sizes)
use_gpu = torch.cuda.is_available()

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def train_model(model, model_test, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # best_model_wts = model.state_dict()
    # best_acc = 0.0
    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch

    if opt.arcface:
        criterion_arcface = losses.ArcFaceLoss(num_classes=opt.nclasses, embedding_size=512)
    if opt.cosface:
        criterion_cosface = losses.CosFaceLoss(num_classes=opt.nclasses, embedding_size=512)
    if opt.circle:
        criterion_circle = CircleLoss(m=0.25, gamma=32)  # gamma = 64 may lead to a better result.
    if opt.triplet:
        miner = miners.MultiSimilarityMiner()
        criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    if opt.lifted:
        criterion_lifted = losses.GeneralizedLiftedStructureLoss(neg_margin=1, pos_margin=0)
    if opt.contrast:
        criterion_contrast = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if opt.sphere:
        criterion_sphere = losses.SphereFaceLoss(num_classes=opt.nclasses, embedding_size=512, margin=4)

    for epoch in range(num_epochs - start_epoch):
        epoch = epoch + start_epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            running_corrects2 = 0.0
            running_corrects3 = 0.0
            # Iterate over data.
            for data, data2, data3, data4 in zip(dataloaders['satellite'], dataloaders['street'], dataloaders['drone'],
                                                 dataloaders['google']):
                # get the inputs
                inputs, labels = data
                inputs2, labels2 = data2
                inputs3, labels3 = data3
                inputs4, labels4 = data4
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    inputs2 = Variable(inputs2.cuda().detach())
                    inputs3 = Variable(inputs3.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                    labels2 = Variable(labels2.cuda().detach())
                    labels3 = Variable(labels3.cuda().detach())
                    if opt.extra_Google:
                        inputs4 = Variable(inputs4.cuda().detach())
                        labels4 = Variable(labels4.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs, outputs2 = model(inputs, inputs2)
                else:
                    if opt.views == 2:
                        outputs, outputs2 = model(inputs, inputs2)
                    elif opt.views == 3:
                        if opt.extra_Google:
                            outputs, outputs2, outputs3, outputs4 = model(inputs, inputs2, inputs3, inputs4)
                        else:
                            outputs, outputs2, outputs3 = model(inputs, inputs2, inputs3)

                return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.lifted or opt.sphere

                if opt.views == 2:
                    _, preds = torch.max(outputs.data, 1)
                    _, preds2 = torch.max(outputs2.data, 1)
                    loss = criterion(outputs, labels) + criterion(outputs2, labels2)
                elif opt.views == 3:
                    if return_feature:
                        logits, ff = outputs
                        logits2, ff2 = outputs2
                        logits3, ff3 = outputs3
                        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                        fnorm2 = torch.norm(ff2, p=2, dim=1, keepdim=True)
                        fnorm3 = torch.norm(ff3, p=2, dim=1, keepdim=True)
                        ff = ff.div(fnorm.expand_as(ff))  # 8*512,tensor
                        ff2 = ff2.div(fnorm2.expand_as(ff2))
                        ff3 = ff3.div(fnorm3.expand_as(ff3))
                        loss = criterion(logits, labels) + criterion(logits2, labels2) + criterion(logits3, labels3)
                        _, preds = torch.max(logits.data, 1)
                        _, preds2 = torch.max(logits2.data, 1)
                        _, preds3 = torch.max(logits3.data, 1)
                        # Multiple perspectives are combined to calculate losses, please join ''--loss_merge'' in run.sh
                        if opt.loss_merge:
                            ff_all = torch.cat((ff, ff2, ff3), dim=0)
                            labels_all = torch.cat((labels, labels2, labels3), dim=0)
                        if opt.extra_Google:
                            logits4, ff4 = outputs4
                            fnorm4 = torch.norm(ff4, p=2, dim=1, keepdim=True)
                            ff4 = ff4.div(fnorm4.expand_as(ff4))
                            loss = criterion(logits, labels) + criterion(logits2, labels2) + criterion(logits3, labels3) +criterion(logits4, labels4)
                            if opt.loss_merge:
                                ff_all = torch.cat((ff_all, ff4), dim=0)
                                labels_all = torch.cat((labels_all, labels4), dim=0)
                        if opt.arcface:
                            if opt.loss_merge:
                                loss += criterion_arcface(ff_all, labels_all)
                            else:
                                loss += criterion_arcface(ff, labels) + criterion_arcface(ff2, labels2) + criterion_arcface(ff3, labels3)  # /now_batch_size
                                if opt.extra_Google:
                                    loss += criterion_arcface(ff4, labels4)  # /now_batch_size
                        if opt.cosface:
                            if opt.loss_merge:
                                loss += criterion_cosface(ff_all, labels_all)
                            else:
                                loss += criterion_cosface(ff, labels) + criterion_cosface(ff2, labels2) + criterion_cosface(ff3, labels3)  # /now_batch_size
                                if opt.extra_Google:
                                    loss += criterion_cosface(ff4, labels4)  # /now_batch_size
                        if opt.circle:
                            if opt.loss_merge:
                                loss += criterion_circle(*convert_label_to_similarity(ff_all, labels_all)) / now_batch_size
                            else:
                                loss += criterion_circle(*convert_label_to_similarity(ff, labels)) / now_batch_size + criterion_circle(*convert_label_to_similarity(ff2, labels2)) / now_batch_size + criterion_circle(*convert_label_to_similarity(ff3, labels3)) / now_batch_size
                                if opt.extra_Google:
                                    loss += criterion_circle(*convert_label_to_similarity(ff4, labels4)) / now_batch_size
                        if opt.triplet:
                            if opt.loss_merge:
                                hard_pairs_all = miner(ff_all, labels_all)
                                loss += criterion_triplet(ff_all, labels_all, hard_pairs_all)
                            else:
                                hard_pairs = miner(ff, labels)
                                hard_pairs2 = miner(ff2, labels2)
                                hard_pairs3 = miner(ff3, labels3)
                                loss += criterion_triplet(ff, labels, hard_pairs) + criterion_triplet(ff2, labels2, hard_pairs2) + criterion_triplet(ff3, labels3, hard_pairs3)# /now_batch_size
                                if opt.extra_Google:
                                    hard_pairs4 = miner(ff4, labels4)
                                    loss += criterion_triplet(ff4, labels4, hard_pairs4)
                        if opt.lifted:
                            if opt.loss_merge:
                                loss += criterion_lifted(ff_all, labels_all)
                            else:
                                loss += criterion_lifted(ff, labels) + criterion_lifted(ff2, labels2) + criterion_lifted(ff3, labels3)  # /now_batch_size
                                if opt.extra_Google:
                                    loss += criterion_lifted(ff4, labels4)
                        if opt.contrast:
                            if opt.loss_merge:
                                loss += criterion_contrast(ff_all, labels_all)
                            else:
                                loss += criterion_contrast(ff, labels) + criterion_contrast(ff2,labels2) + criterion_contrast(ff3, labels3)  # /now_batch_size
                                if opt.extra_Google:
                                    loss += criterion_contrast(ff4, labels4)
                        if opt.sphere:
                            if opt.loss_merge:
                                loss += criterion_sphere(ff_all, labels_all) / now_batch_size
                            else:
                                loss += criterion_sphere(ff, labels) / now_batch_size + criterion_sphere(ff2, labels2) / now_batch_size + criterion_sphere(ff3, labels3) / now_batch_size
                                if opt.extra_Google:
                                    loss += criterion_sphere(ff4, labels4)

                    else:
                        _, preds = torch.max(outputs.data, 1)
                        _, preds2 = torch.max(outputs2.data, 1)
                        _, preds3 = torch.max(outputs3.data, 1)
                        if opt.loss_merge:
                            outputs_all = torch.cat((outputs, outputs2, outputs3), dim=0)
                            labels_all = torch.cat((labels, labels2, labels3), dim=0)
                            if opt.extra_Google:
                                outputs_all = torch.cat((outputs_all, outputs4), dim=0)
                                labels_all = torch.cat((labels_all, labels4), dim=0)
                            loss = 4*criterion(outputs_all, labels_all)
                        else:
                            loss = criterion(outputs, labels) + criterion(outputs2, labels2) + criterion(outputs3, labels3)
                            if opt.extra_Google:
                                loss += criterion(outputs4, labels4)

                # backward + optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    if fp16:  # we use optimier to backward loss
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()
                    ##########
                    if opt.moving_avg < 1.0:
                        update_average(model_test, model, opt.moving_avg)

                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))
                running_corrects2 += float(torch.sum(preds2 == labels2.data))
                if opt.views == 3:
                    running_corrects3 += float(torch.sum(preds3 == labels3.data))

            epoch_loss = running_loss / dataset_sizes['satellite']
            epoch_acc = running_corrects / dataset_sizes['satellite']
            epoch_acc2 = running_corrects2 / dataset_sizes['satellite']

            if opt.views == 2:
                print('{} Loss: {:.4f} Satellite_Acc: {:.4f}  Street_Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc,
                                                                                         epoch_acc2))
            elif opt.views == 3:
                epoch_acc3 = running_corrects3 / dataset_sizes['satellite']
                print('{} Loss: {:.4f} Satellite_Acc: {:.4f}  Street_Acc: {:.4f} Drone_Acc: {:.4f}'.format(phase,
                                                                                                           epoch_loss,
                                                                                                           epoch_acc,
                                                                                                           epoch_acc2,
                                                                                                           epoch_acc3))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'train':
                scheduler.step()
            last_model_wts = model.state_dict()
            if epoch % 20 == 19:
                save_network(model, opt.name, epoch)
            # draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    # save_network(model_test, opt.name+'adapt', epoch)

    return model


######################################################################
# Draw Curve
# ---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model', name, 'train.jpg'))


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.lifted or opt.sphere

if opt.views == 2:
    model = two_view_net(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
                         share_weight=opt.share, circle=return_feature)
elif opt.views == 3:
    model = three_view_net(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
                           share_weight=opt.share, circle=return_feature)

opt.nclasses = len(class_names)

print(model)
# For resume:
if start_epoch >= 40:
    opt.lr = opt.lr * 0.1

ignored_params = list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.1 * opt.lr},
    {'params': model.classifier.parameters(), 'lr': opt.lr}
], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
#
dir_name = os.path.join('./model', name)
if not opt.resume:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    # record every run
    copyfile('train.py', dir_name + '/train.py')
    copyfile('./model.py', dir_name + '/model.py')
    # save opts
    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()
if fp16:
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")

criterion = nn.CrossEntropyLoss()
if opt.moving_avg < 1.0:
    model_test = copy.deepcopy(model)
    num_epochs = 140
else:
    model_test = None
    num_epochs = 120

model = train_model(model, model_test, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=num_epochs)

