# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from torch import nn

import torch
import torch.nn.functional as F


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    label=label.long()
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero(as_tuple =False).squeeze().cuda()
    neg = label.data.eq(0).nonzero(as_tuple =False).squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def DISCLE(pred, target, weight):
    pred_x = (pred[:,:, 0]+pred[:,:, 2])/2
    pred_y = (pred[:,:, 1]+pred[:,:, 3])/2
    pred_w = (-pred[:,:, 0]+pred[:,:, 2])
    pred_h = (-pred[:,:, 1]+pred[:,:, 3])



    target_x = (target[:,:, 0]+target[:,:, 2])/2
    target_y = (target[:,:, 1]+target[:,:, 3])/2
    target_w = (-target[:,:, 0]+target[:,:, 2])
    target_h = (-target[:,:, 1]+target[:,:, 3])
    
    loss=torch.sqrt(torch.pow((pred_x-target_x),2)/target_w+torch.pow((pred_y-target_y),2)/target_h)
    

    weight=weight.view(loss.size())
        
    return  (loss * weight).sum() / (weight.sum()+1e-6)

class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        
        pred_left = pred[:,:, 0]
        pred_top = pred[:,:, 1]
        pred_right = pred[:,:, 2]
        pred_bottom = pred[:,:, 3]

        target_left = target[:,:, 0]
        target_top = target[:,:, 1]
        target_right = target[:,:, 2]
        target_bottom = target[:,:, 3]

        target_aera = (target_right-target_left ) * \
                      (target_bottom-target_top)
        pred_aera = (pred_right-pred_left ) * \
                    (pred_bottom-pred_top)

        w_intersect = torch.min(pred_right, target_right)-torch.max(pred_left, target_left) 
        w_intersect=w_intersect.clamp(min=0)        
        h_intersect = torch.min(pred_bottom, target_bottom) -torch.max(pred_top, target_top)
        h_intersect=h_intersect.clamp(min=0)   
        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        ious=((area_intersect ) / (area_union +1e-6)).clamp(min=0)+1e-6
        
        losses = -torch.log(ious)
        weight=weight.view(losses.size())

            
        return (losses * weight).sum() / (weight.sum()+1e-6)

