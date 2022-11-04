# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import spacy
import pickle
import torch

from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def dependency_dist_func(text, aspect_term):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    # Load spacy's dependency tree into a networkx graph
    edges = []
    
    for token in document:
        # FYI https://spacy.io/docs/api/token
        for child in token.children:
            edges.append((token.i, child.i))
    graph = nx.Graph(edges)

    text_lst = text.split()
    seq_len = len(text_lst)
    text_left, _, _ = text.partition(aspect_term)
    start = len(text_left.split())
    end = start + len(aspect_term.split())
    asp_idx = [i for i in range(start, end)]
    dist_matrix = seq_len*np.ones((seq_len, len(asp_idx))).astype('float32')
    for i, asp in enumerate(asp_idx):
        for j in range(seq_len):
            try:
                dist_matrix[j][i] = nx.shortest_path_length(graph, source=asp, target=j)
            except:
                dist_matrix[j][i] = seq_len/2
    dist_matrix = np.min(dist_matrix, axis=1)
    return dist_matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2dist = {}
    fout = open(filename+'.dist', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        text=text_left+' '+aspect+' '+text_right
        adj_matrix = dependency_dist_func(text, aspect)
        idx2dist[i] = adj_matrix

    pickle.dump(idx2dist, fout)        # 序列化（封装）对象，将对象idx2graph保存到文件fout中去。
    print('done !!!' + filename)
    fout.close()

if __name__ == '__main__':
    process('./datasets/acl-14-short-data/train.raw')
    process('./datasets/acl-14-short-data/test.raw')
    process('./datasets/semeval14/restaurant_train.raw')
    process('./datasets/semeval14/restaurant_test.raw')
    process('./datasets/semeval14/laptop_train.raw')
    process('./datasets/semeval14/laptop_test.raw')
    process('./datasets/semeval15/restaurant_train.raw')
    process('./datasets/semeval15/restaurant_test.raw')
    process('./datasets/semeval16/restaurant_train.raw')
    process('./datasets/semeval16/restaurant_test.raw')
