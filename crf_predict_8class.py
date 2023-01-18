# -*- encoding: utf-8 -*-

import torch
import os
import sys
# sys.path.append('/root/workspace/Bert-BiLSTM-CRF-pytorch')
from utils_8class import tag2idx, idx2tag
from crf_8class import Bert_BiLSTM_CRF
from transformers import T5Tokenizer
from typing import NamedTuple


CRF_MODEL_PATH = '/mnt/8t/jy/HyperGAT_TextClassification-main/HyperGAT_TextClassification-main/Bert-BiLSTM-CRF-pytorch/precision:0.7947336923997607,recall:0.7947336923997607,f1:0.7947336923997607.8.pt'
BERT_PATH = '/mnt/8t/jy/HyperGAT_TextClassification-main/HyperGAT_TextClassification-main/Bert-BiLSTM-CRF-pytorch/Rostlab/prot_t5_xl_uniref50'


class CRF_8(object):
    def __init__(self, crf_model, bert_model):
        self.device = torch.device('cpu')
        self.model = Bert_BiLSTM_CRF(tag2idx)
        self.model.load_state_dict(torch.load(crf_model,map_location=lambda storage, loc: storage))
        # self.model.load(crf_model)
        self.model.to('cpu')
        self.model.eval()
        self.tokenizer = T5Tokenizer.from_pretrained(bert_model)

    def predict(self, text):
        """Using model to predict label
        
        Arguments:
            text {str} -- [description]
        """
        tokens = ['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]']
        # print(tokens)
        xx = self.tokenizer.convert_tokens_to_ids(tokens)
        xx = torch.tensor(xx).unsqueeze(0).to(self.device)
        _, y_hat = self.model(xx,"y")
        pred_tags = []
        for tag in y_hat.squeeze():
            pred_tags.append(idx2tag[tag.item()])
        return pred_tags, tokens

#crf = CRF_8(CRF_MODEL_PATH, BERT_PATH)


def get_crf_and_graph_model(text, crf):
    pred_tags, tokens = crf.predict(text)
    return pred_tags[1:-1]


def predict_pipeline(sequence_list):
    result_list = []
    for value in text:
        current_result = get_crf_and_graph_model(value, model)
        result_list.append(current_result)
        print(current_result)


if __name__ == "__main__":
    model = CRF_8(CRF_MODEL_PATH, BERT_PATH)
    text = ['S S S S']
    for value in text:
        print(get_crf_and_graph_model(value, model))
