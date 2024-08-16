from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


'''注意蒸馏方法为特征蒸馏，学生模型的每一个中间层的特征 是从 一个对应的教师模型学习而来的, 所以 num_teacher = len(trans_feat_s_list) '''
''''''
class CAMKD(nn.Module):
    def __init__(self):
        super(CAMKD, self).__init__()
        # self.crit_ce = nn.CrossEntropyLoss()
        self.crit_ce = nn.CrossEntropyLoss(reduction='none')  #交叉熵损失
        self.crit_mse = nn.MSELoss(reduction='none')  #均方误差损失 不做处理 输出一个与输入相同的张量
        # self.crit_mse = nn.MSELoss(reduction='mean') # 取平均 返回标量

    def forward(self, trans_feat_s_list, mid_feat_t_list, output_feat_t_list, target): 
        # 学生模型的中间特征列表、多教师模型的中间特征列表、多教师模型的输出特征列表、标签
    
        bsz = target.shape[0]


        loss_t = [self.crit_ce(logit_t, target) for logit_t in output_feat_t_list] 
        # 教师模型与标签的交叉熵损失，与中间层无关！以确定教师模型的权重
        num_teacher = len(trans_feat_s_list)
        loss_t = torch.stack(loss_t, dim=0) # 转化成张量 —— "reduction=none"
        weight = (1.0 - F.softmax(loss_t, dim=0)) / (num_teacher - 1)



        loss_st = [] #学生模型损失
        for mid_feat_s, mid_feat_t in zip(trans_feat_s_list, mid_feat_t_list):
            tmp_loss_st = self.crit_mse(mid_feat_s, mid_feat_t).reshape(bsz, -1).mean(-1)
            loss_st.append(tmp_loss_st)
        loss_st = torch.stack(loss_st, dim=0)
        loss = torch.mul(weight, loss_st).sum() # 每个教师模型的损失进行的加权值（weight）
        # loss = torch.mul(attention, loss_st).sum()
        loss /= (1.0*bsz*num_teacher)

        # avg weight
        # loss_st = []
        # for mid_feat_s, mid_feat_t in zip(trans_feat_s_list, mid_feat_t_list):
        #     tmp_loss_st = self.crit_mse(mid_feat_s, mid_feat_t)
        #     loss_st.append(tmp_loss_st)
        # loss_st = torch.stack(loss_st, dim=0)
        # loss = loss_st.mean(0)
        return loss, weight


