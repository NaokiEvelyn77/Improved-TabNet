# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 15:44:52 2021

@author: tiantian05
"""
import torch
import torch.nn as nn

class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)
        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)
        diff_loss = torch.mean(torch.cosine_similarity(input1_l2,input2_l2))
        
        #diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        return diff_loss


# =============================================================================
# 测试函数
# =============================================================================
#    
if __name__=='__main__':
    a=torch.randn(3,5)
    print(a)
    b=torch.randn(3,5)
    print(b)
    loss = DiffLoss()
    print(loss(a,b))
#    
