# -*- encoding: utf-8 -*-

import os
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
from transformers import T5Tokenizer, T5Model

logger = logging.getLogger(__name__)

bert_model = '/mnt/8t/jy/HyperGAT_TextClassification-main/HyperGAT_TextClassification-main/Bert-BiLSTM-CRF-pytorch/Rostlab/prot_t5_xl_uniref50'
# bert_model = '/home/disk1/jy/Rostlab/prot_t5_xl_uniref50'
tokenizer = T5Tokenizer.from_pretrained(bert_model, do_lower_case=False)
# VOCAB = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'B-LOC', 'B-ORG')
# VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-INF', 'I-INF', 'B-PAT', 'I-PAT', 'B-OPS',
#         'I-OPS', 'B-DSE', 'I-DSE', 'B-DRG', 'I-DRG', 'B-LAB', 'I-LAB')
VOCAB=('<PAD>', '[CLS]', '[SEP]', 'H', 'C', 'E')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 200


class NerDataset(Dataset):
    def __init__(self, f_path):
        with open(f_path, 'r', encoding='utf-8') as fr:
            entries = fr.read().strip().split('\n\n')
        sents, tags_li = [], [] # list of lists
        for entry in entries:

            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            # if len(words) > MAX_LEN:
            #     # 先对句号分段
            #     word, tag = [], []
            #     for char, t in zip(words, tags):
            #
            #         if char != '。':
            #             if char != '\ue236':   # 测试集中有这个字符
            #                 word.append(char)
            #                 tag.append(t)
            #         else:
            #             sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
            #             tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
            #             word, tag = [], []
            #     # 最后的末尾
            #     if len(word):
            #         sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
            #         tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
            #         word, tag = [], []
            # else:
            sents.append(["[CLS]"] + words[:MAX_LEN] + ["[SEP]"])
            tags_li.append(['[CLS]'] + tags[:MAX_LEN] + ['[SEP]'])
        self.sents, self.tags_li = sents, tags_li
                

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        x, y = [], []
        is_heads = []
        seq = ""

        for w, t in zip(words, tags):
            seq += w + " " if w not in ("[CLS]", "[SEP]") else w + " "
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)




            # print("xx:",xx)
            # print("tokens:",tokens)
            # print("w:",w)

            # assert len(tokens) == len(xx), f"len(tokens)={len(tokens)}, len(xx)={len(xx)}"

            # 中文没有英文wordpiece后分成几块的情况
            is_head = [1] + [0] * (len(tokens) - 1)
            t = [t] + ['<PAD>'] * (len(tokens) - 1)
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)
        # assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen #句子列表，句子的标签列表，中文就一个字，标签列表，句子长度


    def __len__(self):
        return len(self.sents)
#填充
def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()
    # maxlen=MAX_LEN
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)


    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens


def sov_calculation(observed, prediction):
    # H:3;array0   C:4;array1    E:5array2
    s_list = [[[], []], [[], []], [[], []]]
    s_list_not = [[[], []], [[], []], [[], []]]
    sov_sum = []
    s_list_not_all_type = []
    for type in range(3):
        max_num = []
        min_num = []
        s = [[], []]
        s1 = []
        s2 = []
        type_num = 3 + type
        for index in range(len(observed)):
            if observed[index] == type_num:
                s[0].append(index)
        for index in range(len(prediction)):
            if prediction[index] == type_num:
                s[1].append(index)

        for value in s[0]:
            if len(s1) != 0 and (s1[-1][1] + 1) == value:
                s1[-1][1] = value
            else:
                s1.append([value, value])

        for value in s[1]:
            if len(s2) != 0 and (s2[-1][1] + 1) == value:
                s2[-1][1] = value
            else:
                s2.append([value, value])

        single_1=[True]*len(s1)
        single_2=[True]*len(s2)

        for index_1 in range(len(s1)):
            # single_1.append(True)
            for index_2 in range(len(s2)):
                if s1[index_1][1] >= s2[index_2][0] and s1[index_1][0] <= s2[index_2][1]:
                    s_list[type][0].append(s1[index_1])
                    s_list[type][1].append(s2[index_2])
                    single_1[index_1]=False
                    single_2[index_2]=False

        for index in range(len(single_1)):
            if single_1[index]:
                s_list_not[type][0].append(s1[index])
        for index in range(len(single_2)):
            if single_2[index]:
                s_list_not[type][1].append(s2[index])

        #
        s_list[type][0] = np.unique(np.array(s_list[type][0]), axis=0)
        s_list[type][1] = np.unique(np.array(s_list[type][1]), axis=0)
        s_list_not[type][0] = np.unique(np.array(s_list_not[type][0]), axis=0)
        s_list_not[type][1] = np.unique(np.array(s_list_not[type][1]), axis=0)

        # print("S(i)s1",s_list[type][0])
        # print("S(i)s2",s_list[type][1])
        # print("S'(i)s1",s_list_not[type][0])
        # print("S(i)s2",s_list_not[type][1])

        si_sum = []
        Ni = []
        for value_1 in s_list[type][0]:
            minov = 0
            maxov = 0
            len_s1 = value_1[1] - value_1[0] + 1
            # print(value_1)
            len_s2 = 0
            data_s1_s2 = 0
            for value_2 in s_list[type][1]:
                len_s2 = value_2[1] - value_2[0] + 1
                if value_1[1] >= value_2[0] and value_1[0] <= value_2[1]:
                    Ni.append(len_s1)
                    # print(Ni,len_s1)
                    sov_part_sort = sorted([value_1[0], value_1[1], value_2[0], value_2[1]])
                    # print(sov_part_sort)
                    minov = sov_part_sort[2] - sov_part_sort[1] + 1
                    min_num.append(minov)
                    maxov = sov_part_sort[3] - sov_part_sort[0] + 1
                    max_num.append(maxov)
                    data_s1_s2 = min(maxov - minov, minov, len_s1 // 2, len_s2 // 2)
                    si_sum.append(((minov + data_s1_s2) / maxov) * len_s1)

        s_list_not_type = []
        for value in s_list_not[type][0]:
            s_list_not_type.append(value[1] - value[0] + 1)
        s_list_not_all_type.append(sum(Ni) + sum(s_list_not_type))

        sov_sum.append(sum(si_sum))
        if sum(Ni) != 0:
            if type==0:
                print("SOV(H):",100 * sum(si_sum) / (sum(Ni) + sum(s_list_not_type)))
                print("ACC(H):", 100 * sum(min_num) / sum(max_num))
            elif type==1:
                print("SOV(C):",100 * sum(si_sum) / (sum(Ni) + sum(s_list_not_type)))
                print("ACC(C):", 100 * sum(min_num) / sum(max_num))
            else:
                print("SOV(E):",100 * sum(si_sum) / (sum(Ni) + sum(s_list_not_type)))
                print("ACC(E):", 100 * sum(min_num) / sum(max_num))
    print("SOV:",100 * sum(sov_sum) / sum(s_list_not_all_type))
    return 100 * sum(sov_sum) / sum(s_list_not_all_type)
