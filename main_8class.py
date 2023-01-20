# -*- encoding: utf-8 -*-
import random

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import argparse
from torch.utils import data
from crf_8class import Bert_BiLSTM_CRF
from utils_8class import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag
from layer_dense_classification import Bert_BiLSTM_FC

# from fgsm import FGM
# from torch.optim.lr_scheduler import LambdaLR

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def train_fc(model, iterator, optimizer, criterion, device):
    model.train()  # 启用batch normalization和drop out
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        # print(y)
        # print(x[0].size())
        x = x.to(device)  # 字符id组成的句子
        y = y.to(device)  # 字符的标签的索引

        _y = y  # for monitoring
        optimizer.zero_grad()  # 梯度初始化为0
        loss = model.loss(x, y, criterion)  # logits: (N, T, VOCAB), y: (N, T)

        loss.backward()
        optimizer.step()


def train(model, iterator, optimizer, criterion, device):
    model.train()  # 启用batch normalization和drop out
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        # print(y)
        # print(x[0].size())
        x = x.to(device)  # 字符id组成的句子
        y = y.to(device)  # 字符的标签的索引

        _y = y  # for monitoring
        optimizer.zero_grad()  # 梯度初始化为0
        loss = model.neg_log_likelihood(x, y, criterion)  # logits: (N, T, VOCAB), y: (N, T)
        # print("feats:",feats.size())
        # for index in range(len(feats)):
        #     loss += criterion(feats[index], y[index])
        # logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        # y = y.view(-1)  # (Ns*T,)
        # writer.add_scalar('data/loss', loss.item(), )

        # loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        # scheduler_1.step()

    # 对抗学习的加入

    # fgm = FGM(model, epsilon=1, emb_name='_bert_enc')
    # for i, batch in enumerate(iterator):
    #    words, x, is_heads, tags, y, seqlens = batch
    #    x = x.to(device)#字符id组成的句子
    #    y = y.to(device)#字符的标签的索引
    #    _y = y # for monitoring
    #    optimizer.zero_grad()#F梯度初始化为0
    #    loss = model.neg_log_likelihood(x, y)  # logits: (N, T, VOCAB), y: (N, T)
    #    loss.backward()
    #    fgm.attack()
    #    loss_adv =  model.neg_log_likelihood(x, y)
    #    loss_adv.backward()
    #    fgm.restore()
    #    optimizer.step()

    if i == 0:
        print("=====sanity check======")
        # print("words:", words[0])
        print("x:", x.cpu().numpy()[0][:seqlens[0]])
        # print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
        print("is_heads:", is_heads[0])
        print("y:", _y.cpu().numpy()[0][:seqlens[0]])
        print("tags:", tags[0])
        print("seqlen:", seqlens[0])
        print("=======================")

    if i % 10 == 0:  # monitoring
        print(f"step: {i}, loss: {loss.item()}")


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def eval(model, iterator, f, device):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch
            x = x.to(device)
            # y = y.to(device)

            _, y_hat = model(x, y.to(device))  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
    # #计算句子的准确度
    # count=0
    # for sentence_idx in range(len(Y)):
    #     if Y[sentence_idx]==Y_hat[sentence_idx]:
    #         count+=1
    # sentence_precision=count/len(Y)
    # print("这里预测句子的准确度：",sentence_precision)

    ## gets results and save
    with open("temp", 'w', encoding='utf-8') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    y_true = np.array(
        [tag2idx[line.split()[1]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_pred = np.array(
        [tag2idx[line.split()[2]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    # print(y_true[:20])
    # print(y_pred)
    # {'<PAD>': 0, '[CLS]': 1, '[SEP]': 2, 'H': 3, 'C': 4, 'E': 5}

    num_proposed = len(y_pred[y_pred > 2])
    num_correct = (np.logical_and(y_true == y_pred, y_true > 2)).astype(np.int).sum()
    num_gold = len(y_true[y_true > 2])
    num_all = len(y_pred)

    H_correct = (np.logical_and(y_true == y_pred, y_true == 3)).astype(np.int).sum()
    H_all = len(y_pred[y_pred == 3])

    C_correct = (np.logical_and(y_true == y_pred, y_true == 4)).astype(np.int).sum()
    C_all = len(y_pred[y_pred == 4])

    E_correct = (np.logical_and(y_true == y_pred, y_true == 5)).astype(np.int).sum()
    E_all = len(y_pred[y_pred == 5])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    print(f"num_all:{num_all}")
    print("-" * 20)
    print(f"H_correct:{H_correct}")
    print(f"H_all:{H_all}")
    print("-" * 20)
    print(f"C_correct:{C_correct}")
    print(f"C_all:{C_all}")
    print("-" * 20)
    print(f"E_correct:{E_correct}")
    print(f"E_all:{E_all}")
    try:
        H_accurate = H_correct / H_all
    except ZeroDivisionError:
        H_accurate = 1.0

    try:
        C_accurate = C_correct / C_all
    except ZeroDivisionError:
        C_accurate = 1.0

    try:
        E_accurate = E_correct / E_all
    except ZeroDivisionError:
        E_accurate = 1.0

    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        accurate = num_correct / num_all
    except ZeroDivisionError:
        accurate = 1.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    final = f + ".P%.2f_R%.2f_F%.2f_C%.2f" % (precision, recall, f1, accurate)
    with open(final, 'w', encoding='utf-8') as fout:
        result = open("temp", "r", encoding='utf-8').read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"accurate={accurate}\n")
        fout.write(f"f1={f1}\n")

        # fout.write(f"H_accurate={H_accurate}\n")
        # fout.write(f"C_accurate={C_accurate}\n")
        # fout.write(f"E_accurate={E_accurate}\n")

    os.remove("temp")

    print("precision=%.2f" % precision)
    print("recall=%.2f" % recall)
    print("f1=%.2f" % f1)
    print("accurate=%.2f" % accurate)
    print("H_accurate=%.2f" % H_accurate)
    print("C_accurate=%.2f" % C_accurate)
    print("E_accurate=%.2f" % E_accurate)

    return precision, recall, f1


# 定义获取embbeding的函数
def get_embbeding(model, iterator, str):
    model.eval()
    model.bert.eval()
    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    emb_list = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch
            x = x.to(device)
            # y = y.to(device)

            sentence_emb = model._bert_enc(x)  # y_hat: (N, T)

            # Words.extend(words)
            # Is_heads.extend(is_heads)
            # Tags.extend(tags)
            # Y.extend(y.numpy().tolist())
            # Y_hat.extend(y_hat.cpu().numpy().tolist())
            emb_list.append(np.array(sentence_emb.cpu().numpy().tolist()))
    # 把embbding 存入文件
    pickle_file = open(str, "wb")
    import pickle
    # print(emb_list)
    print(len(emb_list))
    pickle.dump(emb_list, pickle_file)
    print("加载完成！")


if __name__ == "__main__":
    seed_torch()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--emb_batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--fine tuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str,
                        default="checkpoints/8class_crf_linearEveryTransition_200Mluti_6HyperGAT_bert_12_8_batch20_lr_0.00005_1_1")
    parser.add_argument("--classification", type=str, default='crf')
    parser.add_argument("--trainset", type=str, default='./data/train.data.txt')
    parser.add_argument("--validset", type=str, default='./data/test.data.txt')
    # parser.add_argument("--Train", type=str, default='./data/new_train.data.txt')
    # parser.add_argument("--Test", type=str, default='./data/new_test.data.txt')
    # parser.add_argument("--Dev", type=str, default='./data/new_dev.data.txt')
    # parser.add_argument("--local_rank", type=int)
    hp = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if hp.classification == "crf":
        model = Bert_BiLSTM_CRF(tag2idx)
    else:
        model = Bert_BiLSTM_FC(tag2idx)
    # model = nn.DataParallel(model)
    device = torch.device(device)
    model = model.to(device)
    # model.load_state_dict(torch.load('./checkpoints/01/9.pt'))
    print('Initial model Done')
    # model = nn.DataParallel(model)

    train_dataset = NerDataset(hp.trainset)
    eval_dataset = NerDataset(hp.validset)
    # print("train_dataset:",train_dataset)
    # 以下用来获取emb的迭代数据
    # emb_train_dataset=NerDataset(hp.Train)
    # emb_dev_dataset = NerDataset(hp.Dev)
    # emb_test_dataset = NerDataset(hp.Test)

    print('Load Data Done')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss()

    print('Start Train...,')
    # scheduler_1 = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
    result = 0.78
    if hp.classification == "crf":
        for epoch in range(1, hp.n_epochs + 1):  # 每个epoch对dev集进行测试
            # print("train_iter:",train_iter)

            train(model, train_iter, optimizer, criterion, device)

            print(f"=========eval at epoch={epoch}=========")
            if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
            fname = os.path.join(hp.logdir, str(epoch))
            precision, recall, f1 = eval(model, eval_iter, fname, device)

            print("precision, recall, f1:", precision, recall, f1)
            # torch.save(model.state_dict(), f"{fname}.pt")
            # 这里保存微调后的bert模型
            if f1 > result:
                result = f1
                torch.save(model.state_dict(), "precision:" + str(precision) + ",recall:" + str(recall) + ",f1:" +
                           str(f1) + ".8.pt")
                print(
                    "weights were saved to crf_linearEveryTransition_200Mluti_12HyperGAT_bert_12_8_batch20_lr_0.00001_1_1.pt")
    else:
        for epoch in range(1, hp.n_epochs + 1):  # 每个epoch对dev集进行测试
            # print("train_iter:",train_iter)

            train_fc(model, train_iter, optimizer, criterion, device)

            print(f"=========eval at epoch={epoch}=========")
            if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
            fname = os.path.join(hp.logdir, str(epoch))
            precision, recall, f1 = eval(model, eval_iter, fname, device)

    # #训练完后来取embbeding
    # get_embbeding(model, emb_train_iter,'./data_emb/train_emb')
    # get_embbeding(model,emb_dev_iter,'./data_emb/dev_emb')
    # get_embbeding(model, emb_test_iter, './data_emb/test_emb')
