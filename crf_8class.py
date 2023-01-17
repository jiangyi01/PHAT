# -*- encoding: utf-8 -*-

import math

import numpy
import torch
import torch.nn as nn
from transformers import T5Model

import torch.nn.functional as F
from HyperGAT import HGNN_ATT
from HyperUtils import Data, get_KMer_change, get_dict_HyerNodes, get_node_to_sentence_embedding


def trans_to_cuda(variable):

    return variable


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_batch(log_Tensor, axis=-1):  # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0] + \
           torch.log(torch.exp(log_Tensor - torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))


class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, tag_to_ix, hidden_dim=1024):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        # self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=1024, hidden_size=hidden_dim // 2,
                            batch_first=True)
        # self.transitions = nn.Parameter(torch.randn(
        #     self.tagset_size, self.tagset_size
        # ))
        self.hidden_dim = hidden_dim
        self.start_label_id = self.tag_to_ix['[CLS]']
        self.end_label_id = self.tag_to_ix['[SEP]']
        self.fc = nn.Linear(hidden_dim, self.tagset_size)
        # self.fc_hidden_1 = nn.Linear(hidden_dim, (hidden_dim // 16) * 3)
        # self.fc_hidden_2 = nn.Linear((hidden_dim // 16) * 3, (hidden_dim // 32) * 3)
        # self.fc_hidden_3 = nn.Linear((hidden_dim // 32) * 3, (hidden_dim // 64) * 3)
        self.fc_out = nn.Linear(hidden_dim, self.tagset_size)
        self.fc_cat = nn.Linear(self.tagset_size * 3, self.tagset_size)
        self.bert = T5Model.from_pretrained(
            '/mnt/8t/jy/HyperGAT_TextClassification-main/HyperGAT_TextClassification-main/Bert-BiLSTM-CRF-pytorch/Rostlab/prot_t5_xl_uniref50')
        # self.bert = T5Model.from_pretrained('/home/disk1/jy/Rostlab/prot_t5_xl_uniref50')
        self.bert.eval()  # 知用来取bert embedding

        # self.fc_transitions = nn.Linear(hidden_dim, self.tagset_size)
        # self.fc_transitions_hidden_1 = nn.Linear(hidden_dim, (hidden_dim // 16) * 3)
        # self.fc_transitions_hidden_2 = nn.Linear((hidden_dim // 16) * 3, (hidden_dim // 32) * 3)
        # self.fc_transitions_hidden_3 = nn.Linear((hidden_dim // 32) * 3, (hidden_dim // 64) * 3)
        self.fc_transitions_out = nn.Linear(hidden_dim, self.tagset_size * self.tagset_size)
        self.fc_transitions_cat = nn.Linear(self.tagset_size * self.tagset_size, self.tagset_size)

        self.lstm_embedding = nn.LSTM(bidirectional=True, num_layers=4, input_size=2048, hidden_size=1024,
                                      batch_first=True)
        # --------------------------------------------------------
        # 加入超图网络
        self.hgnn_1 = HGNN_ATT(input_size=1024, output_size=1024, n_hid=hidden_dim, dropout=0.5)

        # self.hgnn_2 = HGNN_ATT(input_size=1024, output_size=1024, n_hid=hidden_dim, dropout=0.3)
        #
        # self.hgnn_3 = HGNN_ATT(input_size=1024, output_size=1024, n_hid=hidden_dim, dropout=0.3)

        # self.Ws = nn.parameter.Parameter(torch.randn(1, 1).cuda())

        # self.Wh = nn.parameter.Parameter(torch.randn(1, 1).cuda())
        # self.Wm = nn.parameter.Parameter(torch.randn(1, 1).cuda())
        #
        # self.Wst = nn.parameter.Parameter(torch.randn(1, 1).cuda())
        #
        # self.Wht = nn.parameter.Parameter(torch.randn(1, 1).cuda())

        # self.Wend=torch.randn(1,6).cuda()
        # self.Wend.requires_grad=True
        # self.end_embeds_linear1 = nn.Linear(6, 1)
        # self.end_embeds_linear2 = nn.Linear(512, 256)
        # self.end_embeds_linear3 = nn.Linear(256, 1)

        # self.Ws_2 = torch.randn(1, 1024,requires_grad=True).cuda()
        # self.Wh_2 = torch.randn(1, 1024,requires_grad=True).cuda()
        # torch.randn(1, 1024, requires_grad=True)
        # torch.randn(1, 1024, requires_grad=True)
        # self.layer1 = torch.nn.Linear(2048, 1024)
        # self.layer2 = torch.nn.Linear(1024, 1024)
        # self.hyperEmbedding = nn.parameter.Parameter(torch.rand(30, 1024).cuda())
        # self.hyperEmbedding = nn.Embedding(30, 1024, padding_idx=0)
        self.hyperEmbedding_mer = nn.Embedding(3030, 1024, padding_idx=0)
        # self.hyperEmbedding_mer = nn.parameter.Parameter(torch.rand(3030, 1024).cuda())

        # self.lstm_position_Embedding = nn.parameter.Parameter(torch.rand(200, 1024).cuda())
        # self.lstm_an_Embedding = nn.parameter.Parameter(torch.rand(30, 1024).cuda())
        # --------------------------------------------------------
        # self.transitions.data[self.start_label_id, :] = -10000
        # self.transitions.data[:, self.end_label_id] = -10000
        self.device = torch.device('cpu')
        # self.transitions.to(self.device)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats, lstm_transitions_feats):
        '''
        this also called alpha-recursion or forward recursion, to calculate log_prob of all barX
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(batch_size, 1, self.tagset_size).fill_(-10000.).to(self.device)  # [batch_size, 1, 16]
        # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
        # self.start_label has all of the score. it is log,0 is p=1
        log_alpha[:, 0, self.start_label_id] = 0

        # feats: sentances -> word embedding -> lstm -> MLP -> feats
        # feats is the probability of emission, feat.shape=(1,tag_size)
        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(lstm_transitions_feats[:, t] + log_alpha, axis=-1) + feats[:, t]).unsqueeze(
                1)

        # log_prob of all barX
        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _score_sentence(self, feats, label_ids, lstm_transitions_feats):

        T = feats.shape[1]
        batch_size = feats.shape[0]

        lstm_transitions_feats_size = lstm_transitions_feats.shape[1]
        batch_transitions = lstm_transitions_feats.view(batch_size, lstm_transitions_feats_size,
                                                        self.tagset_size * self.tagset_size)
        # batch_transitions = self.transitions.expand(batch_size, self.tagset_size, self.tagset_size)
        # batch_transitions = batch_transitions.flatten(1)
        print("batch_transitions:", batch_transitions.size())

        score = torch.zeros((feats.shape[0], 1)).to(self.device)
        # the 0th node is start_label->start_word,the probability of them=1. so t begin with 1.
        for t in range(1, T):
            score = score + \
                    batch_transitions[:, t].gather(-1,
                                                   (label_ids[:, t] * self.tagset_size + label_ids[:, t - 1]).view(-1,
                                                                                                                   1)) \
                    + feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)
        return score

    def _bert_enc(self, x):
        """
        x: [batchsize, sent_len]
        enc: [batch_size, sent_len, 1024]
        """
        # print("x:", x.size())
        fraze = True
        if fraze:
            with torch.no_grad():
                embedding = self.bert(input_ids=x, attention_mask=None, decoder_input_ids=x)
            enc = embedding[2]
            # print(enc)
            # print(enc.size())
        else:
            embedding = self.bert(input_ids=x, attention_mask=None, decoder_input_ids=x)
            enc = embedding[2]

        return enc

    def _viterbi_decode(self, feats, lstm_transitions_feats):
        '''
        Max-Product Algorithm or viterbi algorithm, argmax(p(z_0:t|x_0:t))
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # batch_transitions=self.transitions.expand(batch_size,self.tagset_size,self.tagset_size)

        log_delta = torch.Tensor(batch_size, 1, self.tagset_size).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0.

        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.tagset_size), dtype=torch.long)  # psi[0]=0000 useless
        for t in range(1, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(lstm_transitions_feats[:, t] + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T - 2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t + 1].gather(-1, path[:, t + 1].view(-1, 1)).squeeze()

        return max_logLL_allz_allx, path

        # print("feats:",feats.size())
        # print("feats:",feats[1][:])
        # feats=self.endLstm(feats)
        # feats = torch.softmax(feats, dim=2)
        # f_end = self.fc_end(feats[0])
        # f_end = lambda i: self.fc_end(feats[i])

        # f_end=self.fc_end(feats)
        # print("f_end:",f_end.size())
        # f_end=f_end.view(20,f_end.size()[1])
        # print(f_end.size())

        # print("tags:",tags.size())
        # for value in feats:
        #     for val in value:
        #         for i in range(len(val)):
        #             if
        # feats =self.Wend*feats
        # loss=0
        # for index in range(len(feats)):
        #     loss+=criterion(feats[index], tags[index])

        #
        # def _randomInit(self, sentence):
        #     # torch.randn()
        #
        #     random_tensor = torch.ones(len(sentence), len(sentence[0]), 1024)
        #     return random_tensor

    def _get_Hyper_features(self, sent_copy, k, sentence_k, sentence_stride, hgnn):
        merge_sen_mer = get_KMer_change(k, sent_copy, sentence_k, sentence_stride)
        HyperSentence_mer = Data(data=merge_sen_mer)
        alias_inputs_mer, HT_mer, items_mer, node_masks_mer = HyperSentence_mer.get_slice()
        items_mer = trans_to_cuda(torch.Tensor(items_mer).long())
        items_embeds_mer = self.hyperEmbedding_mer(items_mer)
        HT_mer = trans_to_cuda(torch.Tensor(HT_mer).float())
        nodes = hgnn(items_embeds_mer, HT_mer)
        dict_HyerNodes = get_dict_HyerNodes(nodes, items_mer)
        sentence_Hyper_embeds = get_node_to_sentence_embedding(sent_copy, dict_HyerNodes, 2)
        return sentence_Hyper_embeds

    def _get_lstm_HyperGAT_features(self, sentence, tags):
        """sentence is the ids"""

        sent_copy = sentence.cpu().numpy().tolist()
        sentence_Hyper_embeds = self._get_Hyper_features(sent_copy, 2, 12, 8, self.hgnn_1)
        embeds = self._bert_enc(sentence)
        enb_multy = sentence_Hyper_embeds * embeds
        enc, _ = self.lstm(enb_multy)
        # lstm_transitions_feats = self.fc_transitions_hidden_1(enc)
        # lstm_transitions_feats = torch.sigmoid(lstm_transitions_feats)
        # lstm_transitions_feats = self.fc_transitions_hidden_2(lstm_transitions_feats)
        # lstm_transitions_feats = torch.sigmoid(lstm_transitions_feats)
        # lstm_transitions_feats = self.fc_transitions_hidden_3(lstm_transitions_feats)
        # lstm_transitions_feats = torch.sigmoid(lstm_transitions_feats)
        lstm_transitions_feats = self.fc_transitions_out(enc)
        lstm_transitions_feats_out = lstm_transitions_feats.view(enc.size()[0], enc.size()[1],
                                                                 self.tagset_size, self.tagset_size)
        lstm_transitions_feats_cat = torch.sigmoid(lstm_transitions_feats)
        lstm_transitions_feats_cat = self.fc_transitions_cat(lstm_transitions_feats_cat)
        # lstm_feats = self.fc_hidden_1(enc)
        # lstm_feats = torch.sigmoid(lstm_feats)
        # lstm_feats = self.fc_hidden_2(lstm_feats)
        # lstm_feats = torch.sigmoid(lstm_feats)
        # lstm_feats = self.fc_hidden_3(lstm_feats)
        # lstm_feats = torch.sigmoid(lstm_feats)
        lstm_feats = self.fc_out(enc)
        lstm_feats = torch.sigmoid(lstm_feats)
        F.dropout(lstm_feats, 0.3)
        crf_feats = self.fc(enc)
        crf_feats = torch.sigmoid(crf_feats)
        lstm_feats = torch.cat((lstm_feats, crf_feats, lstm_transitions_feats_cat), dim=2)
        lstm_feats = self.fc_cat(lstm_feats)
        return lstm_feats, lstm_transitions_feats_out  # [8, 75, 16]

    def forward(self, sentence, y):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats, lstm_transitions_feats = self._get_lstm_HyperGAT_features(sentence, y)  # [8, 180,768]
        score, tag_seq = self._viterbi_decode(lstm_feats, lstm_transitions_feats)
        return score, tag_seq

        # lstm_feats = self._get_lstm_HyperGAT_features(sentence, y)  # [8, 180,768]
        # lstm_feats_shape = lstm_feats.shape
        # predict_label = torch.zeros([lstm_feats_shape[0], lstm_feats_shape[1]])
        # for index in range(lstm_feats_shape[0]):
        #     for i in range(lstm_feats_shape[1]):
        #         predict_label[index][i] = torch.argmax(lstm_feats[index][i])
        # return "", predict_label

    def neg_log_likelihood(self, sentence, tags, criterion):
        lstm_feats, lstm_transitions_feats = self._get_lstm_HyperGAT_features(sentence,
                                                                              tags)  # [batch_size, max_len, 16]
        forward_score = self._forward_alg(lstm_feats, lstm_transitions_feats)
        gold_score = self._score_sentence(lstm_feats, tags, lstm_transitions_feats)
        # lstm_feats_shape = lstm_feats.shape
        # predict_label = torch.zeros([lstm_feats_shape[0], lstm_feats_shape[1]])
        # for index in range(lstm_feats_shape[0]):
        #     for i in range(lstm_feats_shape[1]):
        #         predict_label[index][i] = lstm_feats[index][i][tags[index][i]]
        # target = tags.view(-1, 1).long()
        # target = target.squeeze()
        # input = lstm_feats.view(-1, 6).float()

        # loss_value = criterion(input=input, target=target)

        return torch.mean(forward_score - gold_score)

    def softmax_likelihood(self, sentence, tags, criterion):
        lstm_feats, lstm_transitions_feats = self._get_lstm_HyperGAT_features(sentence,
                                                                              tags)
        input = lstm_feats.view(-1, 6).float()
        output = torch.nn.Softmax(input, dim=-1)
        target = tags.view(-1, 1).long()
        target = target.squeeze()
        loss = target - output
        return loss
