# -*- coding: utf-8 -*-
# ------------------

# ------------------

import numpy as np
import spacy
import pickle

nlp = spacy.load('en_core_web_sm')


def aspect_position(text):
    # i = 0
    text_list = text.split()
    return len(text_list)


def dependency_adj_matrix(text, aspect, position):
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    text_list = text.split()
    aspect_list = aspect.split()

    for token in document:
        # 赋予方面词的权重
        if str(token) in aspect_list:
            weight = 1
            if token.i < seq_len:
                for j in range(seq_len):
                    if text_list[j] in aspect:
                        sub_weight = 1
                    else:
                        sub_weight = 1 / (abs(j - int(position)) + 1)
                    matrix[token.i][j] += 1 * sub_weight
                    matrix[j][token.i] += 1 * sub_weight
        # 赋予非方面词的权重
        else:
            weight = 1 / (abs(token.i - int(position)) + 1)
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            for child in token.children:
                if str(child) in aspect:
                    weight += 1
                else:
                    weight += 1 / (abs(child.i - int(position)) + 1)
                if child.i < seq_len:
                    matrix[token.i][child.i] += 1 * weight
                    #matrix[child.i][token.i] += 1 * weight

    return matrix


def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename + '.tree_con_a', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        text = text_left + ' ' + aspect + ' ' + text_right
        position = aspect_position(text_left)
        text = text.lower().strip()
        aspect_positions = {}
        aspect_positions[aspect] = position
        adj_matrix = dependency_adj_matrix(text, aspect, position)
        idx2graph[i] = adj_matrix

    pickle.dump(idx2graph, fout)
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
