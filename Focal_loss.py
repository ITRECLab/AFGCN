# -*- coding: utf-8 -*-

from torch import nn
import torch
from torch.nn import functional as F


class focal_loss(nn.Module):
    def __init__(self, alpha=[3, 3, 1], gamma=2, num_classes=3, size_average=True):
        """
        ce_loss 交叉熵损失函数：ce_loss(p,y)=-[log(p)+log(1-p)] =-log(pt) p预测，y真实标签
        focal_loss损失函数, -α(1-pt)**γ *ce_loss(p,y)  ，pt 相当于yi → 1是p, ＜1是1-p
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):  # 执行此句
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):  # preds.size()=tenser([,])  labels.size()=tensor([1,16])
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))  # preds.size(-1)=3 这个-1应该是指倒数第一个维度  preds.size()=tenser([16,3])
        self.alpha = self.alpha.to(preds.device)  # tensor([3., 1., 2.])
        #  dim=1对某一维度的列进行softmax运算   dim=2或-1，是对某一维度的行进行softmax运算
        # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        # ↓就是把 预测值pt归一化了，相加等于1
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)  # torch.log是以自然数e为底的对数函数  log(pt)=-[log(p)+log(1-p)]log(pt)

        #  view(-1,1) 即展开labels操作
        # torch.gather(input待操作数(tensor), dim, index, out=None)  dim(int)待操作的维度
        # index(LongTensor)如何对input进行操作。 out:输出和index维度一致
        # ↓ labels.view(-1,1) → Tensor([16,1]) 代表index。根据index, pt也变成了index维度，值保留的是index位置的值
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 即Pt这部分实现nll_loss( crossempty = log_softmax + nll)
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))  # ce_loss(p,y)= log(pt)
        # X.view(-1)中的-1本意是根据另外一个数来自动调整维度，但是这里只有一个维度，因此就会将X里面的所有维度数据转化成一维的
        self.alpha = self.alpha.gather(0, labels.view(-1))  # 此时的labels维度是tensor([1,16])， α与labels维度同但为浮点数
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())  # -α   ×  (1-pt)**γ *ce_loss(p,y)
        # # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
