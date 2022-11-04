# -*- coding: utf-8 -*-
# ------------------

# ------------------

import numpy as np
import spacy
import pickle

from spacy.tokens import Doc

# WhitespaceTokenizer(), 空格符号分割，就是split(' ') 最简单的一个分词器。
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


def aspect_position(text):
    # i = 0
    text_list = text.split()
    return len(text_list)


def dependency_adj_matrix(text, aspect, position):
    document = nlp(text)
    seq_len = len(text.split())
    # matrix = np.zeros((seq_len, seq_len)).astype('float32')
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(document))

    aspect_list = aspect.split()


    for token in document:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    for token in document:
        weight = 0
        if token.i < seq_len:
            for child in token.children:
                if str(child) in aspect_list:
                    weight += 0.5
                else:
                    weight += 1 / (abs(child.i - int(position)) + 1)
                if child.i < seq_len and matrix[token.i][child.i] != 0 and matrix[child.i][token.i] != 0:
                    matrix[token.i][child.i] += 1 * weight
                    matrix[child.i][token.i] += 1 * weight

    return matrix


def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph_pos', 'wb')
    #graph_idx = 0
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        text = text_left+' '+aspect+' '+text_right
        position = aspect_position(text_left)
        text = text.lower().strip()
        aspect_positions = {}
        aspect_positions[aspect] = position
        adj_matrix = dependency_adj_matrix(text, aspect, position)
        #aspect_graphs[aspect] = adj_matrix
        idx2graph[i] = adj_matrix

    pickle.dump(idx2graph, fout)
    print('done !!!'+filename)
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
