import math

import scipy.sparse as sp
import torch
import numpy as np


class Data():
    def __init__(self, data):
        # inputs = data
        # print(data)
        self.inputs = np.asarray(data,dtype=object)

    def get_slice(self):
        inputs = self.inputs
        # print("inputs:",inputs)
        items, n_node, HT, alias_inputs, node_masks, node_dic = [], [], [], [], [], []

        for u_input in inputs:
            temp_s = []
            # print(u_input)
            for s in u_input:
                # print("s:",s)
                # print("temp_s:",temp_s)
                for value in s:
                    temp_s.append(value)

            temp_l = list(set(temp_s))
            temp_dic = {temp_l[i]: i for i in range(len(temp_l))}
            n_node.append(temp_l)
            alias_inputs.append([temp_dic[i] for i in temp_s])
            node_dic.append(temp_dic)

        max_n_node = np.max([len(i) for i in n_node])

        num_edge = [len(i) for i in inputs]

        max_n_edge = max(num_edge)

        for idx in range(len(inputs)):
            u_input = inputs[idx]
            node = n_node[idx]
            items.append(node + (max_n_node - len(node)) * [0])

            rows = []
            cols = []
            vals = []

            for s in range(len(u_input)):
                for i in np.arange(len(u_input[s])):
                    if u_input[s][i] == 0:
                        continue

                    rows.append(node_dic[idx][u_input[s][i]])
                    cols.append(s)
                    vals.append(1.0)


            u_H = sp.coo_matrix((vals, (rows, cols)), shape=(max_n_node, max_n_edge))
            HT.append(np.asarray(u_H.T.todense()))

            alias_inputs[idx] = [j for j in range(max_n_node)]
            node_masks.append([1 for j in node] + (max_n_node - len(node)) * [0])

        return alias_inputs, HT, items, node_masks


def get_KMer_change(k, sent_copy, sentence_k, sentence_stride):
    cov = sentence_k
    stride = sentence_stride
    sent = []
    merge_sen_mer = []
    if k == 2:

        for index in range(len(sent_copy)):
            sent_length = len(sent_copy[0])
            index_length = math.ceil((sent_length -cov)/ stride) + 1

            # sen_current = []
            merge_sen_mer_current = []
            for i in range(index_length):
                if i < index_length - 1:
                    merge_set = sent_copy[index][i * stride:i * stride + cov]
                    merge_mer = [merge_set[j] * 100 + merge_set[j + 1] for j in range(len(merge_set)) if
                                 j < len(merge_set) - 1]
                    merge_sen_mer_current.append(merge_mer)

                else:
                    merge_set = sent_copy[index][i * stride:]
                    merge_mer = [merge_set[j] * 100 + merge_set[j + 1] for j in range(len(merge_set)) if
                                 j < len(merge_set) - 1]
                    merge_sen_mer_current.append(merge_mer)


            merge_sen_mer.append(merge_sen_mer_current)
        return merge_sen_mer
    else:
        for index in range(len(sent_copy)):
            sent_length = len(sent_copy[0])
            index_length = math.ceil((sent_length -cov)/ stride) + 1
            sen_current = []
            for i in range(index_length):
                if i < index_length - 1:
                    merge_set = sent_copy[index][i * stride:i * stride + cov]
                    sen_current.append(merge_set)

                else:
                    merge_set = sent_copy[index][i * stride:]


                    sen_current.append(merge_set)

            sent.append(sen_current)
        return sent


def get_dict_HyerNodes(nodes, items_mer):
    dict_HyerNodes = {}

    for index in range(len(nodes)):
        for index_d in range(len(nodes[index])):
            dict_HyerNodes[items_mer[index][index_d].cpu().item()] = nodes[index][index_d]

    return dict_HyerNodes


def get_node_to_sentence_embedding(sent_copy, dict_HyerNodes, kmer):
    # print(dict_HyerNodes)
    list_all_stack = []
    if kmer==2:
        for value in sent_copy:
            list_stack = []
            for val in range(len(value)):
                if val < len(value) - 1 and val != 0:
                    v_final_forward = value[val] * 100 + value[val + 1]
                    v_final_back = value[val - 1] * 100 + value[val]
                    if v_final_forward in dict_HyerNodes and v_final_back in dict_HyerNodes:
                        list_stack.append((dict_HyerNodes[v_final_forward] + dict_HyerNodes[v_final_back]) / 2)
                    elif v_final_forward in dict_HyerNodes:
                        list_stack.append(dict_HyerNodes[v_final_forward])
                    elif v_final_back in dict_HyerNodes:
                        list_stack.append(dict_HyerNodes[v_final_back])


                elif val == 0:
                    v_final_forward = value[val] * 100 + value[val + 1]
                    list_stack.append(dict_HyerNodes[v_final_forward])
                elif val == len(value) - 1:
                    v_fina = value[val - 1] * 100 + value[val]
                    if v_fina in dict_HyerNodes:
                        list_stack.append(dict_HyerNodes[v_fina])

            current_set = torch.stack(list_stack, dim=0)
            list_all_stack.append(current_set)
        sentence_Hyper_embeds = torch.stack(list_all_stack, dim=0)
        return sentence_Hyper_embeds
    else:
        for value in sent_copy:
            list_stack = []
            for val in range(len(value)):
                list_stack.append(dict_HyerNodes[value[val]])

            current_set = torch.stack(list_stack, dim=0)
            list_all_stack.append(current_set)
        sentence_Hyper_embeds = torch.stack(list_all_stack, dim=0)
        return sentence_Hyper_embeds