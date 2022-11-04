# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

'''
BucketIterator：相比于标准迭代器，会将类似长度的样本当做一批来处理，
因为在文本处理中经常会需要将每一批样本长度补齐为当前批中最长序列的长度，
因此当样本长度差别较大时，使用BucketIerator可以带来填充效率的提高
'''


class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    # 排序和补齐
    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))  # 向上取整
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_context_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        batch_dependency_tree = []
        batch_dependency_dist = []
        max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            text_indices, context_indices, aspect_indices, left_indices, polarity, dependency_graph, dependency_tree, dependency_dist = \
                item['text_indices'], item['context_indices'], item['aspect_indices'], item['left_indices'], \
                item['polarity'], item['dependency_graph'], item['dependency_tree'], item['dependency_dist']
            text_padding = [0] * (max_len - len(text_indices))
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))
            dependency_dist_padding = [0] * (max_len - len(dependency_dist))
            batch_text_indices.append(text_indices + text_padding)
            batch_context_indices.append(context_indices + context_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_polarity.append(polarity)
            # numpy.pad(array要填补的数组, pad_width, mode填补类型)pad_width是在各维度的各个方向上想要填补的长度,如（1，2），（2，2）），constant模式指定填补的值
            batch_dependency_graph.append(numpy.pad(dependency_graph, \
                                                    (
                                                        (0, max_len - len(text_indices)),
                                                        (0, max_len - len(text_indices))),
                                                    'constant'))
            batch_dependency_tree.append(numpy.pad(dependency_tree, \
                                                   ((0, max_len - len(text_indices)), (0, max_len - len(text_indices))),
                                                   'constant'))
            batch_dependency_dist.append(dependency_dist + dependency_dist_padding)
        return { \
            'text_indices': torch.tensor(batch_text_indices), \
            'context_indices': torch.tensor(batch_context_indices), \
            'aspect_indices': torch.tensor(batch_aspect_indices), \
            'left_indices': torch.tensor(batch_left_indices), \
            'polarity': torch.tensor(batch_polarity), \
            'dependency_graph': torch.tensor(batch_dependency_graph), \
            'dependency_tree': torch.tensor(batch_dependency_tree), \
            'dependency_dist': torch.tensor(batch_dependency_dist)
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
