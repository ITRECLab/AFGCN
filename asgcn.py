# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
import numpy
from math import sqrt


# def AdjDenom(adj):
#     b = adj / adj
#     d=torch.zeros_like(adj)
#     adj1 = torch.where(b / 1 == 1, b, d)
#     denom = torch.sum(adj1, dim=2, keepdim=True)
#     return denom

class DependencyProximity(nn.Module):
    def __init__(self, opt):
        self.opt = opt
        super(DependencyProximity, self).__init__()

    def forward(self, x, aspect_double_idx, text_len, aspect_len, dependency_dist):
        batch_size, seq_len = x.shape[0], x.shape[1]  # torch.Size([64, 14, 600])
        weight = self.weight_matrix(aspect_double_idx, text_len, aspect_len, dependency_dist, batch_size, seq_len).to(self.opt.device)  # torch.Size([64, 14])
        x = weight.unsqueeze(2)*x
        return x

    def weight_matrix(self, aspect_double_idx, text_len, aspect_len, dependency_dist, batch_size, seq_len):
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        dependency_dist = dependency_dist.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-dependency_dist[i,j]/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-dependency_dist[i,j]/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        return torch.tensor(weight)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)  # g * W
        #denom=AdjDenom(adj)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1  # d + 1  '.graph' 用
        denom=denom.int()
        output = torch.matmul(adj, hidden) / denom  # h~ / (d + 1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """

            PairNorm is typically used after each graph convolution operation.
        """
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]

    def forward(self, x):
        if self.mode == 'None':
            return x

        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x

def get_position_ids(max_len):
    position_ids = {}
    position = (max_len - 1) * -1
    position_id = 1
    while position <= max_len - 1:
        position_ids[position] = position_id
        position_id += 1
        position += 1
    return position_ids

class ASGCN(nn.Module):
    def __init__(self, embedding_matrix, opt, norm_mode='None', norm_scale=10):
        super(ASGCN, self).__init__()
        self.opt = opt
        self.hidden_dim1 = opt.hidden_dim
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        # self.w_omega = nn.Parameter(torch.Tensor(opt.hidden_dim * 2, opt.hidden_dim * 2))
        self.w_omega = nn.Parameter(torch.Tensor(1, opt.hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(opt.hidden_dim * 2, 1))
        self.proximity = DependencyProximity(opt)
        self.conv = nn.Conv1d(2 * opt.hidden_dim, 2 * opt.hidden_dim, 3, padding=1)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc3 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc4 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        # self.f_gc1 = GraphConvolution(2 * opt.hidden_dim, opt.hidden_dim)
        # self.f_gc2 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        # self.b_gc1 = GraphConvolution(2 * opt.hidden_dim, opt.hidden_dim)
        # self.b_gc2 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)
        self.norm = PairNorm(norm_mode, norm_scale)

    # 位置权重函数gi=F(hi)=qi*hi
    def position_weight(self, x, aspect_double_idx, text_len, aspect_len, dependency_dist):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        dependency_dist = dependency_dist.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        # 去噪函数部分
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)  # 方面词权重为0
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
                # weight[i].append((1-(j-aspect_double_idx[i,1])/context_len)/2+(1 - dependency_dist[i, j] / context_len)/2)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return weight*x

    # mask 屏蔽掉非方面词的隐藏状态向量，保持方面词状态向量不变
    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return mask*x

    def Position_decay(self, text_out, masks):  # torch.Size([32, 21, 600]) ,torch.Size([32, 21])
        batch_size, max_len = masks.size()
        mask_list = masks.tolist()  # list(32) 元素为float
        mask_list = numpy.array(mask_list)  # list→numpy
        mask_list = mask_list.astype(numpy.int).tolist()  # numpy→list

        for l in mask_list:
            for index, value in enumerate(l):
                if value != 0:
                    l[index] = 1

        len_target = numpy.sum(mask_list, axis=1)  # 每句话中方面词的个数
        position_ids = []
        position_dict = get_position_ids(100)
        for j, sent in enumerate(mask_list):
            position_id = []
            target_id_left = sent.index(1)
            for p in range(max_len):  # 位置衰减函数 L=100, power=γ=1, p=t, target_id_left=i, target_id_left + len_target[j]=j
                if p < target_id_left:  # 方面词左
                    position_id.append(((100 - abs(100 - (position_dict[p - target_id_left]))) / 100) ** 1)
                if target_id_left <= p < target_id_left + len_target[j]:  # 方面词
                    position_id.append(1)
                if target_id_left + len_target[j] <= p <= max_len:  # 方面词右
                    position_id.append(
                        ((100 - abs(100 - (position_dict[p - target_id_left - len_target[j]]))) / 100) ** 1)
            position_ids.append(position_id)
        position_weight = torch.FloatTensor(position_ids).cpu()  # f(t)

        return position_weight

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj, tadj, dependency_dist = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)  # torch.Size([32, 21, 300])
        text_out, (_, _) = self.text_lstm(text, text_len.cpu())  # LSTM 返回out, (ht, ct)  torch.Size([32, 9, 600])


        x = F.relu(self.norm(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len, dependency_dist), adj)))
        x = F.relu(self.norm(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len, dependency_dist), adj)))
        x_t = F.relu(self.norm(self.gc3(self.position_weight(x, aspect_double_idx, text_len, aspect_len, dependency_dist), tadj)))
        x_t = F.relu(self.norm(self.gc4(self.position_weight(x_t, aspect_double_idx, text_len, aspect_len, dependency_dist), tadj)))
        x += 0.2 * x_t

        # f_x = F.relu(self.f_gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        # f_x = F.relu(self.f_gc2(self.position_weight(f_x, aspect_double_idx, text_len, aspect_len), adj))
        # b_x = F.relu(
        #     self.b_gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj.transpose(1, 2)))
        # b_x = F.relu(
        #     self.b_gc2(self.position_weight(b_x, aspect_double_idx, text_len, aspect_len), adj.transpose(1, 2)))
        # x = torch.cat([f_x, b_x], dim=2)

        x = self.mask(x, aspect_double_idx)  # torch.Size([32, 9, 600])

        # 自己加的  临近权重+一维卷积
        text_out = self.proximity(text_out, aspect_double_idx, text_len, aspect_len, dependency_dist).transpose(1,2).to(torch.float32)  # torch.Size([32, 600, 9])
        text_out = F.relu(self.conv(text_out)).transpose(1, 2)  # [(N,Co,L), ...]*len(Ks)  torch.Size([32, 9, 600])


        '''
        # 自己加的 权重衰减函数
        batch_size, max_len, _ = x.size()
        masks = x.sum(axis=-1)  # torch.Size([32, 21])
        pos = self.Position_decay(text_out, masks)  # torch.Size([32, 21])
        # print("这次ok")
        pos = [x.unsqueeze(1).expand(max_len, self.hidden_dim1 * 2) for x in pos]
        pos = torch.stack(pos)  # list → tensor  torch.Size([32, 21, 600])
        text_out = torch.mul(text_out, pos)  # (4) rt=ht*f(t)  torch.Size([32, 9, 600])
        # 一维卷积
        text_out = F.relu(self.conv(text_out.transpose(1, 2))).transpose(1, 2)  # [(N,Co,L), ...]*len(Ks)  torch.Size([32, 9, 600])(batch_size, seq_len, 2 * hidden_dim)
        '''

        # Attention过程
        #u = torch.tanh(torch.matmul(text_out, self.w_omega)) # text_out=x=(batch_size, seq_len, 2 * hidden_dim)  w_omega:(opt.hidden_dim * 2, opt.hidden_dim * 2)
        β =torch.matmul(text_out, x.transpose(1, 2)).sum(2, keepdim=True)  #(batch_size, seq_len, seq_len) → (batch_size, seq_len, 1)
        u = torch.tanh(torch.matmul(β, self.w_omega))
        # u形状是(batch_size, seq_len, 2 * hidden_dim)
        att = torch.matmul(u, self.u_omega)
        # att形状是(batch_size, seq_len, 1)  u_omega:(opt.hidden_dim * 2, 1)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = text_out * att_score
        # scored_x形状是(batch_size, seq_len, 2 * hidden_dim)
        # Attention过程结束
        feat = torch.sum(scored_x, dim=1)  # 加权求和
        # feat形状是(batch_size, 2 * hidden_dim)

        '''
        #alpha_mat = torch.matmul(x, text_out.transpose(1, 2))  # βt= Mask ← *  LSTM ↑
        alpha_mat = torch.tanh(torch.matmul(x, text_out.transpose(1, 2)))  # βt= Mask ← *  LSTM ↑
        # x : torch.Size([32, 21, 600])=(batch_size, seq_len, 2 * hidden_dim)  text_out.t:  torch.Size([32,600,21])
        # alpha_mat: torch.Size([32, 21, 21])
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)  # α
        # alpha_mat.sum: torch.Size([32, 1, 21])  alpha:  torch.Size([32, 1, 21]) (batch_size, 1, seq_len)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim  r= α * LSTM输出
        # torch.Size([32, 1, 600])   x: torch.Size([32, 600])
        '''

        output = self.fc(feat)  # out : torch.Size([32, 3])

        return output