# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F

import pickle
import numpy as np
import os


from fastreid.utils.events import get_event_storage


def log_accuracy(pred_class_logits, gt_classes, topk=(1,)):
    """
    Log the accuracy metrics to EventStorage.
    """
    bsz = pred_class_logits.size(0)
    
    maxk = max(topk)
    _, pred_class = pred_class_logits.topk(maxk, 1, True, True)
    pred_class = pred_class.t()
    correct = pred_class.eq(gt_classes.view(1, -1).expand_as(pred_class))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / bsz))

    storage = get_event_storage()
    storage.put_scalar("cls_accuracy", ret[0])


def cross_entropy_loss(pred_class_outputs, gt_classes, loss_new_weight, eps, alpha=0.2):
    num_classes = pred_class_outputs.size(1)

    if eps >= 0:
        smooth_param = eps
    else:
        # Adaptive label smooth regularization
        soft_label = F.softmax(pred_class_outputs, dim=1)
        smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

    log_probs = F.log_softmax(pred_class_outputs, dim=1)
    
    with torch.no_grad():
        
        targets = torch.ones_like(log_probs)
        targets *= smooth_param / (num_classes - 1)
        targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))
        
        
    loss = (-targets * log_probs).sum(dim=1)
    def del_tensor_ele(arr,index):
      arr1 = arr[0:index]
      arr2 = arr[index+1:]
      return torch.cat((arr1,arr2),dim=0)
      
    
    with torch.no_grad():
     import pickle
     try:
      with open('epoch','rb') as f:
        a_dict2 = pickle.load(f)
        f.close
        if 160 <= a_dict2['epoch'] <=200:
          for i in np.arange(len(loss)):         
            loss[i] = loss[i]*(loss_new_weight[0][i])
          
          
          
        #  try:
         #   with open('want_soft_label.pickle','rb') as f:
          #    a_dict1 = pickle.load(f)
           #   f.close
            #  for i in np.arange(len(loss)):
                
           #     loss[i] = loss[i]*(a_dict1['new_weight'][0][i])
        # except EOFError: 
         #   print ('softlabel  first skip over')
        else:
          pass
     except EOFError: 
      print ('epoch skip over')
    
    
    with torch.no_grad():
        
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)
        
    loss = loss.sum() / non_zero_cnt

    return loss
