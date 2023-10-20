# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import random


from fastreid.config import configurable
from fastreid.layers import *
from fastreid.layers import pooling, any_softmax
from fastreid.layers.weight_init import weights_init_kaiming
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class EmbeddingHead(nn.Module):
    """
    EmbeddingHead perform all feature aggregation in an embedding task, such as reid, image retrieval
    and face recognition

    It typically contains logic to

    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation
    """

    @configurable
    def __init__(
            self,
            *,
            feat_dim,
            embedding_dim,
            num_classes,
            neck_feat,
            pool_type,
            cls_type,
            scale,
            margin,
            with_bnneck,
            norm_type
    ):
        """
        NOTE: this interface is experimental.

        Args:
            feat_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:
        """
        super().__init__()

        # Pooling layer
        assert hasattr(pooling, pool_type), "Expected pool types are {}, " \
                                            "but got {}".format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()

        self.neck_feat = neck_feat

        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck = nn.Sequential(*neck)

        # Classification head
        assert hasattr(any_softmax, cls_type), "Expected cls types are {}, " \
                                               "but got {}".format(any_softmax.__all__, cls_type)
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.bottleneck.apply(weights_init_kaiming)
        nn.init.normal_(self.weight, std=0.01)

    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        scale         = cfg.MODEL.HEADS.SCALE
        margin        = cfg.MODEL.HEADS.MARGIN
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM
        # fmt: on
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'num_classes': num_classes,
            'neck_feat': neck_feat,
            'pool_type': pool_type,
            'cls_type': cls_type,
            'scale': scale,
            'margin': margin,
            'with_bnneck': with_bnneck,
            'norm_type': norm_type
        }

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat[..., 0, 0]

        # Evaluation
        # fmt: off
        if not self.training: return neck_feat
        # fmt: on
        
        
            # get 20 times including noise right label  soft_label
            
        all_new_soft_label = []
        right_probs = []
        t=neck_feat.shape[0]
        v=neck_feat.shape[1]
        #import pdb
        #pdb.set_trace()
        for j in range(20):
            new_neck_feat = neck_feat.clone()
            maskarea=torch.normal(1,0.2,(t,v))
            device = torch.device('cuda:0') 
            maskarea = maskarea.to(device)
            new_neck_feat = new_neck_feat * maskarea
            new_neck_feat = neck_feat.clone()
            
            #nums = np.ones(2048)
            #nums[:300] = 0
            #np.random.shuffle(nums)
            #maskarea = torch.as_tensor(nums)
            #device = torch.device('cuda:0') 
            #maskarea = maskarea.to(device)
            #for i in range(len(new_neck_feat)):
            #  new_neck_feat[i] = new_neck_feat[i] *maskarea
            
            new_logits = F.linear(new_neck_feat, self.weight)
            new_cls_outputs = self.cls_layer(new_logits, targets)
            new_soft_label = F.softmax(new_cls_outputs, dim=1)
            sum_new_soft_label = (new_soft_label).sum(dim=1)
            all_new_soft_label.append(new_soft_label)
            class_num = len(new_soft_label[0])
             
            tensor3 = torch.zeros((t, class_num))
            device = torch.device('cuda:0') 
            tensor3 = tensor3.to(device)
            right_prob_index = tensor3.scatter_(1, targets.unsqueeze(1), 1)
            right_prob = (right_prob_index *new_soft_label).sum(dim=1)
            right_prob2 = (right_prob_index).sum(dim=1)
            right_probs.append(right_prob)
            
            
        
            # get  mean_soft_label
            tensor3 = torch.zeros((1, t))
            device = torch.device('cuda:0') 
            tensor3 = tensor3.to(device)
            
            for i in range(len(right_probs)):
              tensor3 = tensor3+(right_probs[i])
              
            mean_soft_label = tensor3 / (len(right_probs))
            
            #get  variance_soft_label ,first get (m-xi)
            tensor4 = torch.zeros((20, t))
            device = torch.device('cuda:0') 
            tensor4 = tensor4.to(device)
            for i in range(len(right_probs)):
              tensor4[i] = (right_probs[i])-mean_soft_label
              
            all_var_soft_label = (tensor4 *tensor4)
            
            #get  variance_soft_label ,then get sum(square(m-xi))
            tensor5 = torch.zeros((1, t))
            device = torch.device('cuda:0') 
            tensor5 = tensor5.to(device)
            for i in range(len(right_probs)):
              tensor5 = tensor5+(all_var_soft_label[i])  
            
            val_soft_label = tensor5 / (len(all_var_soft_label))
            
            # get contrast of mean_soft_label and val_soft_label
            #con_soft_label = mean_soft_label / (val_soft_label*10000.0)
            con_soft_label = mean_soft_label / (val_soft_label*10000.0)
            
            # get every instance weight
            ri = torch.tanh(1.2*con_soft_label)
            #ri = con_soft_label
            new_weight = torch.zeros((1, t))
            device = torch.device('cuda:0') 
            new_weight = new_weight.to(device)
            batch_softlabel_avg = ri[0].sum()/ len(ri[0])
            #if ri[0][i]>batch_softlabel_avg:
            for i in range(len(ri[0])):
               new_weight[0][i] = (len(ri[0])*ri[0][i]) / ri[0].sum()
            #for i in range(len(ri[0])):
             # if ri[0][i]>batch_softlabel_avg:
              #  new_weight[0][i] = (len(ri[0])*ri[0][i]) / ri[0].sum()
              #else:
               # new_weight[0][i] = 0
            
            #save mean_soft_label and variance_soft_label and weight
            import pickle
            a_dict1 = {'new_weight':new_weight}
            file = open('want_soft_label.pickle','wb')
            pickle.dump(a_dict1,file)
            file.close
            
            
        
            
            

        # Training
        if self.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, self.weight)
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))

        # Pass logits.clone() into cls_layer, because there is in-place operations
        cls_outputs = self.cls_layer(logits.clone(), targets)

        # fmt: off
        if self.neck_feat == 'before':  feat = pool_feat[..., 0, 0]
        elif self.neck_feat == 'after': feat = neck_feat
        else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on
        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": logits.mul(self.cls_layer.s),
            "features": feat,
            "new_weight": new_weight,
        }
