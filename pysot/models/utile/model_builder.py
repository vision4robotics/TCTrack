# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.models.utile.loss import select_cross_entropy_loss,IOULoss,DISCLE
from pysot.models.backbone.temporalbackbone import TemporalAlexNet

from pysot.models.utile.utile import TCT
from pysot.models.utile.utiletest import TCTtest
import matplotlib.pyplot as plt

import numpy as np



class ModelBuilder(nn.Module):
    def __init__(self,label):
        super(ModelBuilder, self).__init__()

        self.backbone = TemporalAlexNet().cuda()
        
        if label=='test':
            self.TCT=TCTtest(cfg).cuda()
        else:
            self.TCT=TCT(cfg).cuda()
        self.cls3loss=nn.BCEWithLogitsLoss()
        self.IOULOSS=IOULoss()

    def template(self, z,x):
        with t.no_grad():
            zf,_,_ = self.backbone.init(z)
            self.zf=zf

            xf,xfeat1,xfeat2 = self.backbone.init(x)   
            
            ppres=self.TCT.conv1(self.xcorr_depthwise(xf,zf))

            self.memory=ppres
            self.featset1=xfeat1
            self.featset2=xfeat2
            

    def xcorr_depthwise(self,x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out
    
    def track(self, x):
        with t.no_grad():
            
            xf,xfeat1,xfeat2 = self.backbone.eachtest(x,self.featset1,self.featset2)  
                        
            loc,cls2,cls3,memory=self.TCT(xf,self.zf,self.memory)
                        
            self.memory=memory
            self.featset1=xfeat1
            self.featset2=xfeat2
            
        return {
                'cls2': cls2,
                'cls3': cls3,
                'loc': loc
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)

        return cls


    def getcentercuda(self,mapp):

        def dcon(x):
           x[t.where(x<=-1)]=-0.99
           x[t.where(x>=1)]=0.99
           return (t.log(1+x)-t.log(1-x))/2 
        
        size=mapp.size()[3]
        #location 
        x=t.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63)-cfg.TRAIN.SEARCH_SIZE//2,size).reshape(-1)).cuda()
        y=t.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63).reshape(-1,1)-cfg.TRAIN.SEARCH_SIZE//2,size).reshape(-1)).cuda()
        
        shap=dcon(mapp)*(cfg.TRAIN.SEARCH_SIZE//2)
        
        xx=np.int16(np.tile(np.linspace(0,size-1,size),size).reshape(-1))
        yy=np.int16(np.tile(np.linspace(0,size-1,size).reshape(-1,1),size).reshape(-1))


        w=shap[:,0,yy,xx]+shap[:,1,yy,xx]
        h=shap[:,2,yy,xx]+shap[:,3,yy,xx]
        x=x-shap[:,0,yy,xx]+w/2+cfg.TRAIN.SEARCH_SIZE//2
        y=y-shap[:,2,yy,xx]+h/2+cfg.TRAIN.SEARCH_SIZE//2

        anchor=t.zeros((cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.NUM_GPU,size**2,4)).cuda()

        anchor[:,:,0]=x-w/2
        anchor[:,:,1]=y-h/2
        anchor[:,:,2]=x+w/2
        anchor[:,:,3]=y+h/2
        return anchor
    

    
    def forward(self,data):
        """ only used in training
        """

        presearch=data['pre_search'].cuda()
        template = data['template'].cuda()
        search =data['search'].cuda()
        bbox=data['bbox'].cuda()
        labelcls2=data['label_cls2'].cuda()
        labelxff=data['labelxff'].cuda()
        labelcls3=data['labelcls3'].cuda()
        weightxff=data['weightxff'].cuda()
        
        
        presearch=t.cat((presearch,search.unsqueeze(1)),1)    
            
        zf = self.backbone(template.unsqueeze(1))
        
        xf = self.backbone(presearch) ###b l c w h
        xf=xf.view(cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.NUM_GPU,cfg.TRAIN.videorange+1,xf.size(-3),xf.size(-2),xf.size(-1))
        
        loc,cls2,cls3=self.TCT(xf[:,-1,:,:,:],zf,xf[:,:-1,:,:,:].permute(1,0,2,3,4))

       
        cls2 = self.log_softmax(cls2) 

        
 
        cls_loss2 = select_cross_entropy_loss(cls2, labelcls2)
        cls_loss3 = self.cls3loss(cls3, labelcls3)  
        
        pre_bbox=self.getcentercuda(loc) 
        bbo=self.getcentercuda(labelxff) 
        
        loc_loss1=self.IOULOSS(pre_bbox,bbo,weightxff)
        loc_loss2=DISCLE(pre_bbox,bbo,weightxff)
        loc_loss=cfg.TRAIN.w2*loc_loss1+cfg.TRAIN.w3*loc_loss2

        
        cls_loss=cfg.TRAIN.w4*cls_loss2+cfg.TRAIN.w5*cls_loss3

        

        outputs = {}
        outputs['total_loss'] =\
            cfg.TRAIN.LOC_WEIGHT*loc_loss\
                +cfg.TRAIN.CLS_WEIGHT*cls_loss
                
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss1'] = loc_loss1
        outputs['loc_loss2'] = loc_loss2
                                                    #2 4 1  都用loss2

        return outputs
