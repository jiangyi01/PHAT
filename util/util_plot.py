import collections
import numpy as np
import os
from time import time

import torch
import seaborn as sns
import umap
import pandas as pd
import shap
import xgboost

from sklearn import manifold, datasets
from sklearn.preprocessing import StandardScaler

from matplotlib.ticker import NullFormatter
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.transforms as ts
import matplotlib.ticker as ticker

from util import util_data

plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
flag = True

colors = {1: ['#81d4fa'],
          2: ['#81d4fa', '#ef9a9a'],
          3: ['#7AB4D0', '#7D9DD2', '#ef9a9a'],
          4: ['#7AB4D0', '#B7D16F', '#9683D6', '#ef9a9a'],
          5: ['#CA8B66', '#7AB4D0', '#CFCB6C', '#e52d5d', '#9683D6'],
          6: ['#C86A63', '#CA8B66', '#CCAB69', '#CFCB6C', '#B7D16F', '#7D9DD2'],
          7: ['#C86A63', '#CA8B66', '#CCAB69', '#CFCB6C', '#B7D16F', '#87DEBA', '#87B9DE'],
          8: ['#DE8787', '#DEB487', '#D4DE87', '#9FDE87', '#87DEBA', '#87B9DE', '#8C87DE', '#C587DE'],
          9: ['#e52d5d', '#e01b77', '#d31b92', '#bb2cad', '#983fc5', '#7b60e0', '#9FDE87', '#87DEBA', '#87B9DE'],
          10: ['#DE8787', '#DEB487', '#D4DE87', '#9FDE87', '#87DEBA', '#87B9DE', '#8C87DE', '#547bf3', '#0091ff',
               '#00b6ff'],
          11: ['#C86A63', '#CA8B66', '#CCAB69', '#CFCB6C', '#B7D16F', '#87B9DE', '#8C87DE', '#e01b77', '#d31b92',
               '#bb2cad', '#983fc5'],
          12: ['#e52d5d', '#e01b77', '#d31b92', '#bb2cad', '#983fc5', '#7b60e0', '#547bf3', '#0091ff', '#00b6ff',
               '#00d0da',
               '#00df83', '#a4e312'],
          }
image_type = {
    'all_need': ['draw_umap', 'draw_shap', 'draw_ROC_PRC_curve', 'draw_negative_density', 'draw_positive_density',
                 'draw_tra_ROC_PRC_curve', 'draw_result_histogram','epoch_plot'],
    '1_in_3': {'prot': 'draw_hist_image', 'DNA': 'draw_dna_hist_image', 'RNA': 'draw_rna_hist_image'},
    False: ['draw_dna_rna_prot_length_distribution_image']}

# image_type={'all_need':['draw_umap', 'draw_shap', 'draw_ROC_PRC_curve','draw_negative_density','draw_positive_density','draw_tra_ROC_PRC_curve'],
#             '1_in_3':{'prot':'draw_hist_image','DNA':'draw_dna_hist_image','RNA':'draw_rna_hist_image'},
#             False:['draw_dna_rna_prot_length_distribution_image']}

def draw_hist_image(train_data, test_data, config):
    # train_data 和 test_data 要保证有sequences 和 labels成员属性
    keyvalueP = {}
    keyvalueF = {}
    keyvalueP['A'] = 0
    keyvalueP['R'] = 0
    keyvalueP['N'] = 0
    keyvalueP['D'] = 0
    keyvalueP['C'] = 0
    keyvalueP['Q'] = 0
    keyvalueP['E'] = 0
    keyvalueP['G'] = 0
    keyvalueP['H'] = 0
    keyvalueP['I'] = 0
    keyvalueP['L'] = 0
    keyvalueP['K'] = 0
    keyvalueP['M'] = 0
    keyvalueP['F'] = 0
    keyvalueP['P'] = 0
    keyvalueP['S'] = 0
    keyvalueP['T'] = 0
    keyvalueP['W'] = 0
    keyvalueP['Y'] = 0
    keyvalueP['V'] = 0

    keyvalueF['A'] = 0
    keyvalueF['R'] = 0
    keyvalueF['N'] = 0
    keyvalueF['D'] = 0
    keyvalueF['C'] = 0
    keyvalueF['Q'] = 0
    keyvalueF['E'] = 0
    keyvalueF['G'] = 0
    keyvalueF['H'] = 0
    keyvalueF['I'] = 0
    keyvalueF['L'] = 0
    keyvalueF['K'] = 0
    keyvalueF['M'] = 0
    keyvalueF['F'] = 0
    keyvalueF['P'] = 0
    keyvalueF['S'] = 0
    keyvalueF['T'] = 0
    keyvalueF['W'] = 0
    keyvalueF['Y'] = 0
    keyvalueF['V'] = 0

    sequeces = train_data[0]
    labels = train_data[1]
    for i in range(len(sequeces)):
        seq = sequeces[i]
        label = labels[i]
        if (label == 1):  # 正例
            for word in seq:
                if word not in keyvalueP.keys():
                    keyvalueP[word] = 1
                else:
                    keyvalueP[word] += 1
        else:
            for word in seq:
                if word not in keyvalueF.keys():
                    keyvalueF[word] = 1
                else:
                    keyvalueF[word] += 1
    fig, ax = plt.subplots()
    title = "The number of different protein residues(Train)"
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(title)
    plt.xlabel("The types of protein residues", fontsize='x-large', fontweight='light')
    plt.ylabel("The number of protein residues", fontsize='x-large', fontweight='light')
    # 拿到数据
    Px, Py, Nx, Ny = list(keyvalueP.keys()), list(keyvalueP.values()), \
                     list(keyvalueF.keys()), list(keyvalueF.values())
    # 根据坐标画柱状图
    ind = np.arange(20) + 20
    width = 0.4
    plt.bar(ind, Py, color=colors[2][0], edgecolor='#929195', width=width, alpha=1.0, label="Positive")
    plt.bar(ind + width, Ny, color=colors[2][1], edgecolor='#929195', width=width, alpha=0.50, label="Negative")
    plt.legend(loc="upper left", )
    plt.legend(frameon=False)
    plt.xticks(np.arange(20) + 20 + 0.2, ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
                                          'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                                          'T', 'W', 'Y', 'V'),
               fontsize=11)
    plt.savefig('{}/{}.{}'.format(config['savepath'], 'prot_train_statistics', 'png'))
    # plt.show()

    keyvalueP['A'] = 0
    keyvalueP['R'] = 0
    keyvalueP['N'] = 0
    keyvalueP['D'] = 0
    keyvalueP['C'] = 0
    keyvalueP['Q'] = 0
    keyvalueP['E'] = 0
    keyvalueP['G'] = 0
    keyvalueP['H'] = 0
    keyvalueP['I'] = 0
    keyvalueP['L'] = 0
    keyvalueP['K'] = 0
    keyvalueP['M'] = 0
    keyvalueP['F'] = 0
    keyvalueP['P'] = 0
    keyvalueP['S'] = 0
    keyvalueP['T'] = 0
    keyvalueP['W'] = 0
    keyvalueP['Y'] = 0
    keyvalueP['V'] = 0

    keyvalueF['A'] = 0
    keyvalueF['R'] = 0
    keyvalueF['N'] = 0
    keyvalueF['D'] = 0
    keyvalueF['C'] = 0
    keyvalueF['Q'] = 0
    keyvalueF['E'] = 0
    keyvalueF['G'] = 0
    keyvalueF['H'] = 0
    keyvalueF['I'] = 0
    keyvalueF['L'] = 0
    keyvalueF['K'] = 0
    keyvalueF['M'] = 0
    keyvalueF['F'] = 0
    keyvalueF['P'] = 0
    keyvalueF['S'] = 0
    keyvalueF['T'] = 0
    keyvalueF['W'] = 0
    keyvalueF['Y'] = 0
    keyvalueF['V'] = 0

    sequeces = test_data[0]
    labels = test_data[1]
    for i in range(len(sequeces)):
        seq = sequeces[i]
        label = labels[i]
        if (label == 1):  # 正例
            for word in seq:
                if word not in keyvalueP.keys():
                    keyvalueP[word] = 1
                else:
                    keyvalueP[word] += 1
        else:
            for word in seq:
                if word not in keyvalueF.keys():
                    keyvalueF[word] = 1
                else:
                    keyvalueF[word] += 1
    fig, ax = plt.subplots()
    title = "The number of different protein residues(Test)"
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(title)
    plt.xlabel("The types of protein residues", fontsize='x-large', fontweight='light')
    plt.ylabel("The number of protein residues", fontsize='x-large', fontweight='light')
    # 拿到数据
    Px, Py, Nx, Ny = list(keyvalueP.keys()), list(keyvalueP.values()), \
                     list(keyvalueF.keys()), list(keyvalueF.values())
    # 根据坐标画柱状图
    ind = np.arange(20) + 20
    width = 0.4
    plt.bar(ind, Py, color=colors[2][0], edgecolor='#929195', width=width, alpha=1.0, label="Positive")
    plt.bar(ind + width, Ny, color=colors[2][1], edgecolor='#929195', width=width, alpha=0.50, label="Negative")
    plt.legend(loc="upper left", )
    plt.legend(frameon=False)
    plt.xticks(np.arange(20) + 20 + 0.2, ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
                                          'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                                          'T', 'W', 'Y', 'V'),
               fontsize=11)
    plt.savefig('{}/{}.{}'.format(config['savepath'], 'prot_test_statistics', 'png'))
    # plt.show()


def draw_dna_hist_image(train_data, test_data, config):
    keyvalueP = {}
    keyvalueF = {}
    keyvalueP['A'] = 0
    keyvalueP['T'] = 0
    keyvalueP['C'] = 0
    keyvalueP['G'] = 0

    keyvalueF['A'] = 0
    keyvalueF['T'] = 0
    keyvalueF['C'] = 0
    keyvalueF['G'] = 0

    sequeces = train_data[0]
    labels = train_data[1]

    for i in range(len(sequeces)):
        seq = sequeces[i]
        label = labels[i]

        if (label == 1):  # 正例
            for word in seq:
                if word not in keyvalueP.keys():
                    keyvalueP[word] = 1
                else:
                    keyvalueP[word] += 1
        else:
            for word in seq:
                if word not in keyvalueF.keys():
                    keyvalueF[word] = 1
                else:
                    keyvalueF[word] += 1
    fig, ax = plt.subplots()
    title = "The number of different base(Train)"
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(title)
    plt.xlabel("The types of base", fontsize='x-large', fontweight='light')
    plt.ylabel("The number of base", fontsize='x-large', fontweight='light')
    # 拿到数据
    Px, Py, Nx, Ny = list(keyvalueP.keys()), list(keyvalueP.values()), \
                     list(keyvalueF.keys()), list(keyvalueF.values())
    # 根据坐标画柱状图
    ind = np.arange(4) + 20
    width = 0.4
    plt.bar(ind, Py, color=colors[2][0], edgecolor='#929195', width=width, alpha=1.0, label="Positive")
    plt.bar(ind + width, Ny, color=colors[2][1], edgecolor='#929195', width=width, alpha=0.50, label="Negative")
    plt.legend(loc="upper left", )
    plt.legend(frameon=False)
    plt.xticks(np.arange(4) + 20 + 0.2, ('A', 'T', 'C', 'G'),
               fontsize=11)
    plt.savefig('{}/{}.{}'.format(config['savepath'], 'dna_train_statistics', 'png'))
    # plt.show()

    keyvalueP = {}
    keyvalueF = {}
    keyvalueP['A'] = 0
    keyvalueP['T'] = 0
    keyvalueP['C'] = 0
    keyvalueP['G'] = 0

    keyvalueF['A'] = 0
    keyvalueF['T'] = 0
    keyvalueF['C'] = 0
    keyvalueF['G'] = 0

    sequeces = test_data[0]
    labels = test_data[1]

    for i in range(len(sequeces)):
        seq = sequeces[i]
        label = labels[i]

        if (label == 1):  # 正例
            for word in seq:
                if word not in keyvalueP.keys():
                    keyvalueP[word] = 1
                else:
                    keyvalueP[word] += 1
        else:
            for word in seq:
                if word not in keyvalueF.keys():
                    keyvalueF[word] = 1
                else:
                    keyvalueF[word] += 1
    fig, ax = plt.subplots()
    title = "The number of different base(Test)"
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(title)
    plt.xlabel("The types of base", fontsize='x-large', fontweight='light')
    plt.ylabel("The number of base", fontsize='x-large', fontweight='light')
    # 拿到数据
    Px, Py, Nx, Ny = list(keyvalueP.keys()), list(keyvalueP.values()), \
                     list(keyvalueF.keys()), list(keyvalueF.values())
    # 根据坐标画柱状图
    ind = np.arange(4) + 20
    width = 0.4
    plt.bar(ind, Py, color=colors[2][0], edgecolor='#929195', width=width, alpha=1.0, label="Positive")
    plt.bar(ind + width, Ny, color=colors[2][1], edgecolor='#929195', width=width, alpha=0.50, label="Negative")
    plt.legend(loc="upper left", )
    plt.legend(frameon=False)
    plt.xticks(np.arange(4) + 20 + 0.2, ('A', 'T', 'C', 'G'),
               fontsize=11)
    plt.savefig('{}/{}.{}'.format(config['savepath'], 'dna_test_statistics', 'png'))
    # plt.show()


def draw_rna_hist_image(train_data, test_data, config):
    keyvalueP = {}
    keyvalueF = {}
    keyvalueP['A'] = 0
    keyvalueP['U'] = 0
    keyvalueP['C'] = 0
    keyvalueP['G'] = 0

    keyvalueF['A'] = 0
    keyvalueF['U'] = 0
    keyvalueF['C'] = 0
    keyvalueF['G'] = 0

    sequeces = train_data[0]
    labels = train_data[1]

    for i in range(len(sequeces)):
        seq = sequeces[i]
        label = labels[i]

        if (label == 1):  # 正例
            for word in seq:
                if word not in keyvalueP.keys():
                    keyvalueP[word] = 1
                else:
                    keyvalueP[word] += 1
        else:
            for word in seq:
                if word not in keyvalueF.keys():
                    keyvalueF[word] = 1
                else:
                    keyvalueF[word] += 1
    fig, ax = plt.subplots()
    title = "The number of different base(Train)"
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(title)
    plt.xlabel("The types of base", fontsize='x-large', fontweight='light')
    plt.ylabel("The number of base", fontsize='x-large', fontweight='light')
    # 拿到数据
    Px, Py, Nx, Ny = list(keyvalueP.keys()), list(keyvalueP.values()), \
                     list(keyvalueF.keys()), list(keyvalueF.values())
    # 根据坐标画柱状图
    ind = np.arange(4) + 20
    width = 0.4
    plt.bar(ind, Py, color=colors[2][0], edgecolor='#929195', width=width, alpha=1.0, label="Positive")
    plt.bar(ind + width, Ny, color=colors[2][1], edgecolor='#929195', width=width, alpha=0.50, label="Negative")
    plt.legend(loc="upper left", )
    plt.legend(frameon=False)
    plt.xticks(np.arange(4) + 20 + 0.2, ('A', 'U', 'C', 'G'),
               fontsize=11)
    plt.savefig('{}/{}.{}'.format(config['savepath'], 'rna_train_statistics', 'png'))
    # plt.show()

    keyvalueP = {}
    keyvalueF = {}
    keyvalueP['A'] = 0
    keyvalueP['U'] = 0
    keyvalueP['C'] = 0
    keyvalueP['G'] = 0

    keyvalueF['A'] = 0
    keyvalueF['U'] = 0
    keyvalueF['C'] = 0
    keyvalueF['G'] = 0

    sequeces = test_data.sequences
    labels = test_data.labels

    for i in range(len(sequeces)):
        seq = sequeces[i]
        label = labels[i]

        if (label == 1):  # 正例
            for word in seq:
                if word not in keyvalueP.keys():
                    keyvalueP[word] = 1
                else:
                    keyvalueP[word] += 1
        else:
            for word in seq:
                if word not in keyvalueF.keys():
                    keyvalueF[word] = 1
                else:
                    keyvalueF[word] += 1
    fig, ax = plt.subplots()
    title = "The number of different base(Test)"
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(title)
    plt.xlabel("The types of base", fontsize='x-large', fontweight='light')
    plt.ylabel("The number of base", fontsize='x-large', fontweight='light')
    # 拿到数据
    Px, Py, Nx, Ny = list(keyvalueP.keys()), list(keyvalueP.values()), \
                     list(keyvalueF.keys()), list(keyvalueF.values())
    # 根据坐标画柱状图
    ind = np.arange(4) + 20
    width = 0.4
    plt.bar(ind, Py, color=colors[2][0], edgecolor='#929195', width=width, alpha=1.0, label="Positive")
    plt.bar(ind + width, Ny, color=colors[2][1], edgecolor='#929195', width=width, alpha=0.50, label="Negative")
    plt.legend(loc="upper left", )
    plt.legend(frameon=False)
    plt.xticks(np.arange(4) + 20 + 0.2, ('A', 'U', 'C', 'G'),
               fontsize=11)
    plt.savefig('{}/{}.{}'.format(config['savepath'], config.learn_name, 'rna_test_statistics', 'png'))
    # plt.show()


# def draw_dna_rna_prot_length_distribution_image(train_data, test_data, config):
#     # train_data 中是处理之后的训练集中的正负例样本长度集合
#     # test_data 中是处理之后的测试集中的正负例样本长度集合
#     # 或者在这个方法里用循环处理数据
#     # train_data 列表 01
#     train_positive_lengths = []
#     train_negative_lengths = []
#     test_positive_lengths = []
#     test_negative_lengths = []
#     train_sequences = train_data[0]
#     train_labels = train_data[1]
#     test_seauences = test_data[0]
#     test_labels = test_data[1]
#     for i in range(len(train_labels)):
#         if train_labels[i] == 1:
#             train_positive_lengths.append(len(train_sequences[i]))
#         else:
#             train_negative_lengths.append(len(train_sequences[i]))
#     data1 = train_positive_lengths
#     data2 = train_negative_lengths
#     xlabel = 'Length'
#     ylabel = 'Number'
#     fig, ax = plt.subplots()
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.legend(loc="upper left", )
#     plt.title('The length of ' + config['type'] + '\'s sequences(Train)')
#     plt.xlabel(xlabel, fontsize='x-large', fontweight='light')
#     plt.ylabel(ylabel, fontsize='x-large', fontweight='light')
#     plt.hist(data1, bins=10, edgecolor='black', alpha=1, color=colors[2][0], histtype='bar',
#              align='mid', orientation='vertical', rwidth=None, log=False, label='positive', stacked=False, )
#     plt.hist(data2, bins=10, edgecolor='black', alpha=0.50, color=colors[2][1], histtype='bar',
#              align='mid', orientation='vertical', rwidth=None, log=False, label='negative', stacked=False, )
#     plt.legend(frameon=False)
#     plt.savefig('{}/{}.{}'.format(config['savepath'], config['type'] + '_train_length_statistics', 'jpg'))
#     # plt.show()
#
#     for i in range(len(test_labels)):
#         if test_labels[i] == 1:
#             test_positive_lengths.append(len(test_seauences[i]))
#         else:
#             test_negative_lengths.append(len(test_seauences[i]))
#     data1 = test_positive_lengths
#     data2 = test_negative_lengths
#     xlabel = 'Length'
#     ylabel = 'Number'
#     fig, ax = plt.subplots()
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.legend(loc="upper left", )
#
#     plt.title('The length of' + config['type'] + '\'s sequences(Test)')
#     plt.xlabel(xlabel, fontsize='x-large', fontweight='light')
#     plt.ylabel(ylabel, fontsize='x-large', fontweight='light')
#     plt.hist(data1, bins=10, edgecolor='black', alpha=1, color=colors[2][0], histtype='bar',
#              align='mid', orientation='vertical', rwidth=None, log=False, label='positive', stacked=False, )
#     plt.hist(data2, bins=10, edgecolor='black', alpha=0.50, color=colors[2][1], histtype='bar',
#              align='mid', orientation='vertical', rwidth=None, log=False, label='negative', stacked=False, )
#     plt.legend(frameon=False)
#     plt.savefig('{}/{}.{}'.format(config['savepath'], config['type'] + '_test_length_statistics', 'jpg'))
#     # plt.show()

def draw_dna_rna_prot_length_distribution_image(train_data, test_data, config):
    # train_data 中是处理之后的训练集中的正负例样本长度集合
    # test_data 中是处理之后的测试集中的正负例样本长度集合
    # 或者在这个方法里用循环处理数据
    # train_data 列表 01
    train_positive_lengths = []
    train_negative_lengths = []
    test_positive_lengths = []
    test_negative_lengths = []
    train_sequences = train_data[0]
    train_labels = train_data[1]
    test_seauences = test_data[0]
    test_labels = test_data[1]
    length_data = {'pos': {}, 'neg': {}}
    pos = []
    neg = []

    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            train_positive_lengths.append(len(train_sequences[i]))
        else:
            train_negative_lengths.append(len(train_sequences[i]))
    data1 = train_positive_lengths
    data2 = train_negative_lengths
    xlabel = 'Length'
    ylabel = 'Number'

    for data in data1:
        if data in length_data['pos']:
            length_data['pos'][data] = length_data['pos'][data] + 1
        else:
            length_data['pos'][data] = 1
    for data in data2:
        if data in length_data['neg']:
            length_data['neg'][data] = length_data['pos'][data] + 1
        else:
            length_data['neg'][data] = 1
    # print(data1)
    # print(data2)
    lengths = data1
    for i in data2:
        lengths.append(i)
    print(lengths)
    lengths = list(set(lengths))
    for length in lengths:
        if length in length_data['pos']:
            pos.append(length_data['pos'][length])
        else:
            pos.append(0)
        if length in length_data['neg']:
            neg.append(length_data['neg'][length])
        else:
            neg.append(0)

    x = np.arange(len(lengths))  # the label locations
    # print(x,pos,neg)
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width / 2, pos, width, color=colors[2][0], label='positive')
    rects2 = ax.bar(x + width / 2, neg, width, color=colors[2][1], label='negative')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Length')
    ax.set_ylabel('Number')
    ax.set_title('The length of ' + config['type'] + '\'s sequences(Train)')
    ax.set_xticks(x)
    ax.set_xticklabels(lengths)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('{}/{}.{}'.format(config['savepath'], config['type'] + '_train_length_statistics', 'png'))
    # plt.show()
    pos = []
    neg = []
    for i in range(len(test_labels)):
        if test_labels[i] == 1:
            test_positive_lengths.append(len(test_seauences[i]))
        else:
            test_negative_lengths.append(len(test_seauences[i]))
    data1 = test_positive_lengths
    data2 = test_negative_lengths
    xlabel = 'Length'
    ylabel = 'Number'
    for data in data1:
        if data in length_data['pos']:
            length_data['pos'][data] = length_data['pos'][data] + 1
        else:
            length_data['pos'][data] = 1
    for data in data2:
        if data in length_data['neg']:
            length_data['neg'][data] = length_data['pos'][data] + 1
        else:
            length_data['neg'][data] = 1
    lengths = data1
    for i in data2:
        lengths.append(i)
    lengths = list(set(lengths))
    for length in lengths:
        if length in length_data['pos']:
            pos.append(length_data['pos'][length])
        else:
            pos.append(0)
        if length in length_data['neg']:
            neg.append(length_data['neg'][length])
        else:
            neg.append(0)

    x = x = np.arange(len(lengths))  # the label locations
    width = 0.35  # the width of the bars
    # print(x,pos,neg)
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, pos, width, color=colors[2][0], label='positive')
    rects2 = ax.bar(x + width / 2, neg, width, color=colors[2][1], label='negative')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Length')
    ax.set_ylabel('Number')
    ax.set_title('The length of ' + config['type'] + '\'s sequences(Train)')
    ax.set_xticks(x)
    ax.set_xticklabels(lengths)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('{}/{}.{}'.format(config['savepath'], config['type'] + '_test_length_statistics', 'png'))
    # plt.show()


def draw_ROC_PRC_curve(roc_datas, prc_datas, config):
    # roc_data = [FPR, TPR, AUC]
    # prc_data = [recall, precision, AP]

    f, (ax1, ax2) = plt.subplots(figsize=(16, 8), ncols=2)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    lw = 2

    plt.subplot(1, 2, 1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    colorlocal = colors[len(roc_datas)]

    for index, roc_data in enumerate(roc_datas):
        y = util_data.smooth(roc_data[1])
        # y_smooth = make_interp_spline(np.array(roc_data[0]), np.array(roc_data[1]))(x_smooth)
        plt.plot(roc_data[0], y, color=colorlocal[index], lw=lw,
                 label=config['names'][index] + ' (AUC = %0.3f)' % roc_data[2])  ### 假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel('False Positive Rate',labelpad=4.5, fontdict={'weight': 'normal', 'size': 16})
    plt.ylabel('True Positive Rate',labelpad=8.5, fontdict={'weight': 'normal', 'size': 16})
    plt.title('ROC curve', fontdict={'weight': 'normal', 'size': 20})
    leg = plt.legend(frameon=False, loc="lower right", prop={'weight': 'normal', 'size': 16})
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

    plt.subplot(1, 2, 2)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # plt.step(self.prc_data[0], self.prc_data[1], color='b', alpha=0.2,where='post')
    # plt.fill_between(prc_data[0], prc_data[1], step='post', alpha=0.2,color='b')
    for index, prc_data in enumerate(prc_datas):
        y = util_data.smooth(prc_data[1])
        plt.plot(prc_data[0], y, color=colorlocal[index],
                 lw=lw, label=config['names'][index] + ' (AP = %0.3f)' % prc_data[2])  ### 假正率为横坐标，真正率为纵坐标做曲线


    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel('Recall',labelpad=4.5, fontdict={'weight': 'normal', 'size': 16})
    plt.ylabel('Precision', labelpad=8.5, fontdict={'weight': 'normal', 'size': 16})
    # plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve', fontdict={'weight': 'normal', 'size': 20})
    leg = plt.legend(frameon=False, loc="lower left", prop={'weight': 'normal', 'size': 16})
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

    plt.savefig(
        '{}/{}.{}'.format(config['savepath'], 'ROC_PRC', 'png'))
    # plt.show()


def draw_tra_ROC_PRC_curve(roc_datas, prc_datas, config):
    # roc_data = [FPR, TPR, AUC]
    # prc_data = [recall, precision, AP]

    # roc_data = [FPR, TPR, AUC]
    # prc_data = [recall, precision, AP]

    f, (ax1, ax2) = plt.subplots(figsize=(16, 8), ncols=2)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    lw = 2

    colorlocal = colors[len(roc_datas)]

    plt.subplot(1, 2, 1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    for index, roc_data in enumerate(roc_datas):
        y = util_data.smooth(roc_data[1])
        # y_smooth = make_interp_spline(np.array(roc_data[0]), np.array(roc_data[1]))(x_smooth)
        plt.plot(roc_data[0], y, color=colorlocal[index], lw=lw,
                 label=config['tra_name'][index] + ' (AUC = %0.3f)' % roc_data[2])  ### 假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel('False Positive Rate',labelpad=4.5, fontdict={'weight': 'normal', 'size': 16})
    plt.ylabel('True Positive Rate',labelpad=8.5, fontdict={'weight': 'normal', 'size': 16})
    plt.title('ROC curve', fontdict={'weight': 'normal', 'size': 20})
    leg = plt.legend(frameon=False, loc="lower right", prop={'weight': 'normal', 'size': 16})
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

    plt.subplot(1, 2, 2)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # plt.step(self.prc_data[0], self.prc_data[1], color='b', alpha=0.2,where='post')
    # plt.fill_between(prc_data[0], prc_data[1], step='post', alpha=0.2,color='b')
    for index, prc_data in enumerate(prc_datas):
        y = util_data.smooth(prc_data[1])
        plt.plot(prc_data[0], y, color=colorlocal[index],
                 lw=lw, label=config['tra_name'][index] + ' (AP = %0.3f)' % prc_data[2])  ### 假正率为横坐标，真正率为纵坐标做曲线

    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel('Recall',labelpad=4.5, fontdict={'weight': 'normal', 'size': 16})
    plt.ylabel('Precision', labelpad=8.5, fontdict={'weight': 'normal', 'size': 16})
    # plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve', fontdict={'weight': 'normal', 'size': 20})
    leg = plt.legend(frameon=False, loc="lower left", prop={'weight': 'normal', 'size': 16})
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

    plt.savefig(
        '{}/{}.{}'.format(config['savepath'], 'tra_ROC_PRC', 'png'))
    # plt.show()


# def draw_statistics_bar(traindataset, testdataset, config):
#     totaldataset = []
#     totaldataset.extend(traindataset)
#     totaldataset.extend(testdataset)
#
#     plt.figure()
#
#     if config.model == 'DNAbert':
#         colors = ['#4DE199', '#F4E482', '#BAAD4E','#827919']
#         statstic = [0,0,0,0] # A, C, T, G
#         labels = ['A', 'C', 'T', 'G']
#         for seq in totaldataset:
#             for i in range(len(seq)):
#                 if seq[i] == 'A':
#                     statstic[0] = statstic[0] + 1
#                 elif seq[i] == 'C':
#                     statstic[1] = statstic[1] + 1
#                 elif seq[i] == 'T':
#                     statstic[2] = statstic[2] + 1
#                 elif seq[i] == 'G':
#                     statstic[3] = statstic[3] + 1
#         # print(statstic)
#         plt.bar(labels, statstic, color=colors)  # or `color=['r', 'g', 'b']`
#     elif config.model == 'prot_bert_bfd' or config.model == 'prot_bert':
#         colors = ['#e52d5d', '#e01b77', '#d31b92', '#bb2cad', '#983fc5', '#7b60e0', '#547bf3', '#0091ff', '#00b6ff', '#00d0da', '#00df83', '#a4e312' ]
#         statstic = collections.defaultdict(int)
#         for seq in totaldataset:
#             for i in seq:
#                 statstic[i] += 1
#         # print(statstic)
#         labels = statstic.keys()
#         plt.bar(statstic.keys(), statstic.values(), color=colors[:len(labels)])  # or `color=['r', 'g', 'b']`
#
#     plt.savefig('{}/{}/{}.{}'.format(config.path_save, config.learn_name, 'statistics', config.save_figure_type))
#
#     plt.show()

def draw_umap(repres_list, label_list, config):
    # 配色、名称、坐标轴待统一样式修改
    # print(repres_list)
    # print(np.array(repres_list).shape)
    # fig=plt.figure()
    cmap = ListedColormap(['#00beca', '#f87671'])
    repres = np.array(repres_list)
    label = np.array(label_list)
    scaled_data = StandardScaler().fit_transform(repres)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(scaled_data)
    colors = np.array(["#00beca", "#f87671"])
    # print(embedding)
    sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=label, cmap=cmap, s=5
    )
    # l1, = plt.plot([], [], 'o', color='#00beca', label='positive')
    # l2, = plt.plot([], [], 'o', color='#f87671', label='negative')
    # global flag
    # if flag:
    #     plt.legend(bbox_to_anchor=(-0.15, 1.1), loc=1, borderaxespad=0)
    #     flag=False
    # plt.legend(loc='best')

    # fig, ax = plt.subplots()
    # # title = "The number of different protein residues(Train)"
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.gca().set_aspect('equal', 'datalim')

    cbar = plt.colorbar(sc, ticks=[0, 1])
    cbar.ax.set_yticklabels(['pos', 'neg'])  # horizontal colorbar

    # plt.title('UMAP projection ', fontsize=24)
    # plt.show()


def draw_shap(repres_list, label_list, config):
    # 这里需要计算一个dataframe 表格
    # 这里假设 n * M 的n 是样本
    repres_list = pd.DataFrame(repres_list)
    # print(repres_list.shape)
    # repres_list.columns = [str(i) for i in range(len(repres_list[0]))]
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(repres_list, label=label_list), 100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(repres_list)
    shap.summary_plot(shap_values, repres_list, show=False)


def draw_negative_density(repres_list, label_list, config):
    plt.figure(figsize=(30, 15))
    fig, ax = plt.subplots()
    # mid = [[0.2,0.3,0.4,0.5,0.6,0.7]]
    len_ls = np.array(repres_list)
    y_ls = np.array(label_list)
    # sns.kdeplot(len_ls[y_ls == 0], shade=True, alpha=.25, label='model A', common_norm=False, color='#CC3366')
    # sns.kdeplot(len_ls1[y_ls == 0], shade=True, alpha=.25, label='model B', color='#00CCCC')
    for i in range(config['model_number']):
        mid = np.array(repres_list[i])
        sns.kdeplot(mid[label_list[i] == 1], shade=True, alpha=.25, label=config['names'][i],
                    color=colors[config['model_number']][i])
    ax.vlines(0.25, 0, 0.8, colors='#999999', linestyles="dashed")
    ax.vlines(0.75, 0, 1.2, colors='#999999', linestyles="dashed")
    ax.tick_params(direction='out', labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([])
    # ax.set_xticks([-0.25, 0.05, 0.35, 0.65, 0.95, 1.25])
    # ax.set_xticklabels(('0.0', '0.2', '0.4', '0.6', '0.8', '1.0'))
    # ax.set_ylim(0, 1.4)
    plt.ylabel('Density (predicted negative samples)', fontsize=12)
    plt.xlabel('Confidence', fontsize=12)

    plt.legend(bbox_to_anchor=(0.1, 1.4), loc=2, borderaxespad=0, numpoints=1, fontsize=12, frameon=False)
    fig.subplots_adjust(top=0.7, left=0.2)
    plt.tight_layout()
    plt.savefig('{}/{}.{}'.format(config['savepath'], 'negative_density', 'png'))
    # plt.show()


def draw_positive_density(repres_list, label_list, config):
    plt.figure(figsize=(30, 15))
    fig, ax = plt.subplots()
    # sns.kdeplot(len_ls[y_ls == 0], shade=True, alpha=.25, label='model A', common_norm=False, color='#CC3366')
    # sns.kdeplot(len_ls1[y_ls == 0], shade=True, alpha=.25, label='model B', color='#00CCCC')
    for i in range(config['model_number']):
        mid = np.array(repres_list[i])
        sns.kdeplot(mid[label_list[i] == 0], shade=True, alpha=.25, label=config['names'][i],
                    color=colors[config['model_number']][i])

    ax.vlines(0.25, 0, 0.8, colors='#999999', linestyles="dashed")
    ax.vlines(0.75, 0, 1.2, colors='#999999', linestyles="dashed")
    ax.tick_params(direction='out', labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([])
    # ax.set_xticklabels(('0.0', '0.2', '0.4', '0.6', '0.8', '1.0'))
    # ax.set_ylim(0, 1.4)
    plt.ylabel('Density (predicted positive samples)', fontsize=12)
    plt.xlabel('Confidence', fontsize=12)

    plt.legend(bbox_to_anchor=(0.1, 1.4), loc=2, borderaxespad=0, numpoints=1, fontsize=12, frameon=False)
    fig.subplots_adjust(top=0.7, left=0.2)
    plt.tight_layout()
    plt.savefig('{}/{}.{}'.format(config['savepath'], 'positive_density', 'png'))
    # plt.show()


def epoch_plot(epoch_data, num_of_moudel, config):
    step_log_interval = epoch_data['step_log_interval']
    train_metric_record = epoch_data['train_metric_record']
    train_loss_record = epoch_data['train_loss_record']
    step_test_interval = epoch_data['step_test_interval']
    test_metric_record = epoch_data['test_metric_record']
    test_loss_record = epoch_data['test_loss_record']

    plt.figure(22, figsize=(16, 12))
    # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(16, 8), ncols=2, nrows=2)
    f, (ax1, ax2) = plt.subplots(figsize=(16, 8), ncols=2)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)


    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # ax3.spines['right'].set_visible(False)
    # ax3.spines['top'].set_visible(False)
    # ax4.spines['right'].set_visible(False)
    # ax4.spines['top'].set_visible(False)

    colorlocal = colors[int(num_of_moudel)]

    for i in range(num_of_moudel):
        # ax1.plot(step_log_interval[i], train_metric_record[i], colors = colorlocal[i], label=config['names'][i])
        # ax2.plot(step_log_interval[i], train_loss_record[i], colors = colorlocal[i], label=config['names'][i])
        test_metric_record_y = util_data.smooth(test_metric_record[i])
        test_loss_record_y = util_data.smooth(test_loss_record[i])
        ax1.plot(step_test_interval[i], test_metric_record_y, color = colorlocal[i], label=config['names'][i])
        ax2.plot(step_test_interval[i], test_loss_record_y, color = colorlocal[i], label=config['names'][i])
    # ax1.title("Train Acc Curve", fontsize=23)
    # ax1.xlabel("Step", fontsize=20)
    # ax1.ylabel("Accuracy", fontsize=20)
    #
    # ax2.title("Train Loss Curve", fontsize=23)
    # ax2.xlabel("Step", fontsize=20)
    # ax2.ylabel("Loss", fontsize=20)

    ax1.set_title("Test Acc Curve", fontsize=18)
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Accuracy", fontsize=14)

    ax2.set_title("Test Loss Curve", fontsize=18)
    ax2.set_xlabel("Step", fontsize=14)
    ax2.set_ylabel("Loss", fontsize=14)

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)

    plt.savefig('{}/{}.{}'.format(config['savepath'], 'epoch_plot', 'png'))


def draw_result_histogram(data, data_temp, config):
    # ACC, Sensitivity, Specificity, AUC, MCC
    labels = ['ACC', 'Sensitivity', 'Specificity', 'AUC', 'MCC']

    width = 0.2
    x = np.arange(len(labels)) * 2

    plt.figure(figsize=(8, 8))
    len_data = len(data)
    colorlocal = colors[len_data]

    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for index, bardata in enumerate(data):
        ax.bar(x + width * index, bardata, width, label=config['names'][index], color=colorlocal[index])

    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_xticks(x + (len_data / 2 - 0.5) * width)
    ax.set_xticklabels(labels)
    plt.xticks(size=12)
    leg = ax.legend(bbox_to_anchor=(0.1, 1.4), loc=2, borderaxespad=0, numpoints=1, fontsize=12, frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

    fig.tight_layout()  # 自动调整子图充满整个屏幕
    plt.savefig('{}/{}.{}'.format(config['savepath'], 'result_histogram', 'png'))
    # plt.show()


def construct_data(data):
    datas = {'all_need': [],
             '1_in_3': [data['train_data'], data['test_data']],
             False: [data['train_data'], data['test_data']]}
    # for i in range()
    draw_umap = [data['repres_list'], data['label_list']]
    datas['all_need'].append(draw_umap)
    draw_shap = [data['repres_list'], data['label_list']]
    datas['all_need'].append(draw_shap)
    draw_ROC_PRC_curve = [data['roc_datas'], data['prc_datas']]
    datas['all_need'].append(draw_ROC_PRC_curve)

    neg = np.array(data['neg_list'])
    label = np.array(data['label_list'])
    draw_negative_density = [neg, label]
    datas['all_need'].append(draw_negative_density)

    pos = np.array(data['pos_list'])
    draw_positive_density = [pos, label]
    datas['all_need'].append(draw_positive_density)

    draw_tra_ROC_PRC_curve = [data['roc_datas'] + data['tra_roc_datas'], data['prc_datas'] + data['tra_prc_datas']]
    datas['all_need'].append(draw_tra_ROC_PRC_curve)

    draw_result_histogram = [data['best_performance'], data['config']]
    datas['all_need'].append(draw_result_histogram)

    epoch_plot = [data['epoch_data'], data['config']['model_number']]
    datas['all_need'].append(epoch_plot)

    # data['config']['savepath'] = './data/result/train01'
    config = data['config']

    return datas, config


def draw_plots(data, config):
    # 为了保证函数的优雅性，需要将画图函数的参数统一修改为data1,data2,config三个
    # 把每个函数的plt.show()取消掉
    # 测试数据中config参数不全&train\testFile data格式不对
    # 传参给此函数时data为字典：{type:[[[data1],[data2]],[[data1],[data2]]...]}
    # 问题：柱状图为什么plt.show()了两次

    tag = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    if config['model_number'] % 2 == 0:
        col = 2
    else:
        col = 3

    # print(config['model_number'])
    # fig1 = plt.figure()
    if config['model_number'] == 2 or config['model_number'] == 3:
        fig1 = plt.figure(figsize=(10, 3))
        # print(2,config['model_number'])
    else:
        fig1 = plt.figure()
        # print('hhhh')

    if config['model_number'] == 1:
        # pass
        eval(image_type['all_need'][0])(data['all_need'][0][0][0], data['all_need'][0][1][0], config)
        # l1, = plt.plot([], [], 'o', color='#00beca', label='positive')
        # l2, = plt.plot([], [], 'o', color='#f87671', label='negative')
        # plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
        plt.savefig('{}/{}.{}'.format(config['savepath'], 'UMAP', 'png'))
        plt.figure()
        eval(image_type['all_need'][1])(data['all_need'][1][0][0], data['all_need'][1][1][0], config)

        plt.savefig('{}/{}.{}'.format(config['savepath'], 'SHAP', 'png'))
        # plt.show()

    else:
        for i in range(config['model_number']):

            ax = fig1.add_subplot(int(config['model_number'] / col), col, i + 1)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            eval(image_type['all_need'][0])(data['all_need'][0][0][i], data['all_need'][0][1][i], config)
            # if i == 2:
            # fig1.set_figheight(10)
            # fig1.set_figwidth(3)
            # plt.axis('square')

            # ax.title.set_text(config['names'][type_list[i]])
            ax.set_title(config['names'][i])

            trans = ts.ScaledTranslation(-20 / 72, 7 / 72, fig1.dpi_scale_trans)
            if config['model_number'] == 2:
                ax.text(0.05, 1.0, tag[i], transform=ax.transAxes + trans, fontweight='bold')
            else:
                ax.text(0.1, 1.0, tag[i], transform=ax.transAxes + trans, fontweight='bold')

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)

        # l1, = plt.plot([], [], 'o', color='#00beca', label='positive')
        # l2, = plt.plot([], [], 'o', color='#f87671', label='negative')
        # plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
        # plt.tight_layout()
        plt.savefig('{}/{}.{}'.format(config['savepath'], 'UMAP', 'png'))

        # shap plot
        fig2 = plt.figure()
        for i in range(config['model_number']):
            # ax = fig.add_subplot(1, 3, i + 1)
            ax = fig2.add_subplot(int(config['model_number'] / col), col, i + 1)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            eval(image_type['all_need'][1])(data['all_need'][1][0][i], data['all_need'][1][1][i], config)
            ax.set_xlabel('SHAP value of ' + config['names'][i])
            trans = ts.ScaledTranslation(-20 / 72, 7 / 72, fig2.dpi_scale_trans)
            ax.text(-0.1, 1.0, tag[i], transform=ax.transAxes + trans, fontweight='bold',
                    fontdict={ 'size': 14 })
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.1, hspace=0.3)
        plt.savefig('{}/{}.{}'.format(config['savepath'], 'SHAP', 'png'))
        # plt.show()
    # 其他都需要画的图
    for img in range(2, len(image_type['all_need'])):
        eval(image_type['all_need'][img])(data['all_need'][img][0], data['all_need'][img][1], config)

    # 3选1
    # print(data['1_in_3'][0])
    eval(image_type['1_in_3'][config['type']])(data['1_in_3'][0], data['1_in_3'][1], config)
    # 画motif还是长度分布
    if config['if_same']:
        print("start plot motif")
        motif_title = ['Motif statistics of the positives (Train)', 'Motif statistics of the negatives (Train)',
                       'Motif statistics of the positives (Test)', 'Motif statistics of the negatives (Test)']
        for i in range(4):
            motif = "/home/weilab/anaconda3/envs/wy/bin/weblogo --resolution 500 --format PNG -f " + \
                    config['fasta_list'][i] + " -o " + config['savepath'] + "/motif_" + (str)(
                i) + ".png" + " --title " + " ' " + motif_title[i] + " ' "
            os.system(motif)
    else:
        eval(image_type[False][0])(data['1_in_3'][0], data['1_in_3'][1], config)


def plot_interface(plot_data):
    data, config = construct_data(plot_data)
    draw_plots(data, config)


if __name__ == '__main__':
    plot_data = torch.load('prot_test.pth')
    # plot_data['config']['tra_name'] = ['a', 'b', 'c']
    # # print(plot_data['config'])
    # # print(plot_data['repres_list'])
    # # print(len(plot_data['repres_list'][0][0]))
    # data, config = construct_data(plot_data)
    # draw_plots(data, config)
    # print(plot_data['config'])
    print(plot_data['config']['names'])
    plot_data['config']['savepath'] = './data/result/train03/plot'
    # print(plot_data['train_data'])
    # print('__________')
    # print(plot_data['test_data'])
    # draw_hist_image(plot_data['train_data'],plot_data['test_data'],plot_data['config'])
    data, config = construct_data(plot_data)
    draw_plots(data, config)
    # draw_negative_density(plot_data['neg_list'],plot_data['label_list'],plot_data['config'])
    # draw_positive_density(plot_data['pos_list'], plot_data['label_list'], plot_data['config'])
