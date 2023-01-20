# -*- coding: utf-8 -*-
# @Time    : 2022/4/6 19:12
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: protdefault.py

from transformers import BertModel, BertTokenizer
import re, torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import requests
from util import util_file

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

def delete_duplicate(seq):
    seqsort=[]
    for i in seq:
        if i not in seqsort:
            seqsort.append(i)
    seqstr={}
    for s in seqsort:
        seqstr[s[:-1]]=s[-1:]
    return seqstr
aa_dict = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', \
           'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', \
           'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'SEC': 'U', 'PLY': 'O'}

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()

        global max_len, d_model
        d_model = 1024
        self.tokenizer = BertTokenizer.from_pretrained('/home/weilab/molecular_analysis_server/biology_python/pretrain/prot_bert_bfd', do_lower_case=False)
        self.bert = BertModel.from_pretrained("/home/weilab/molecular_analysis_server/biology_python/pretrain/prot_bert_bfd")
        # freeze_bert(self)
        self.cnn = nn.Sequential(nn.Conv1d(in_channels=1024,
                                        out_channels=1024,
                                        kernel_size=13,
                                           stride=1,
                                           padding=6),
                              nn.ReLU(),
                              # nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
                                 )
        self.conv1d = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(1, 1024), stride=(1, 1),
                                    padding=(0, 0)),
                                  nn.ReLU())


        self.q = nn.Parameter(torch.empty(d_model,))
        self.q.data.fill_(1)
        self.block1 = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.block2 = nn.Sequential(
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def attention(self, input, q):
        # x = q*input
        # att_weights = F.softmax(x, 2)
        att_weights = F.softmax(q, 0)
        output = torch.mul(att_weights, input)
        return output
    def forward(self, input_seq):
        input_seq = ' '.join(input_seq)
        input_seq = re.sub(r"[UZOB]", "X", input_seq)
        encoded_input = self.tokenizer(input_seq, return_tensors='pt')
        for key in encoded_input:
            encoded_input[key] = encoded_input[key].cuda()
        output = self.bert(**encoded_input)
        output = output[0]
        representation = output.view(-1, 1024)
        representation = self.block1(representation)
        return representation

    def get_logits(self, x):
        with torch.no_grad():
            output = self.forward(x)
        logits = self.block2(output)
        return logits

def get_index_aa_dic(pdbid, chain_id, config):
	# whole_dict = {}
	# for pid in querys:
	sequence = []
	address = '/data/result/proteinFunction/' + config.learn_name + '/pdb_file/' + pdbid + '.pdb'
	chain = chain_id
	with open(address, 'r') as f:
		for line in f:
			line = line.split()
			if 'ATOM' in line[0] or 'HETATM' in line[0] or 'TER' in line[0]:
				if 'HETATM' in line[0] and len(line[0]) > 6:
					if line[3] == chain:
						index = line[4]
					elif line[3][0] == chain:
						index = line[3][1:]
					else:
						continue
					amino = line[2]
					if amino in aa_dict:
						amino = aa_dict[amino]
					else:
						amino = 'X'
					sequence.append(index + amino)
				else:
					if line[0] == 'TER':
						continue
					amino = line[3]
					if line[4] == chain:
						index = line[5]
					elif line[4][0] == chain:
						index = line[4][1:]
					else:
						continue
					if amino in aa_dict:
						amino = aa_dict[amino]
					else:
						amino = 'X'
					sequence.append(index + amino)
		stringdict = delete_duplicate(sequence)
	return stringdict

def main(config, choiceid):

	PDBid, chain, test_dataset = util_file.load_PDB_fasta(config.path_data)
	detectType = ['DNA', 'RNA', 'Peptide']
	model = BERT().cuda()

	if int(choiceid) == 0:
		model.load_state_dict(torch.load('/home/weilab/molecular_analysis_server/biology_python/pretrain/prot_annotation/PepBCL.pl')['model'])
	elif int(choiceid) == 1:
		model.load_state_dict(torch.load('/home/weilab/molecular_analysis_server/biology_python/pretrain/prot_annotation/PepBCL.pl')['model'])
	elif int(choiceid) == 2:
		model.load_state_dict(torch.load('/home/weilab/molecular_analysis_server/biology_python/pretrain/prot_annotation/PepBCL.pl')['model'])

	# model_dict = torch.load('PepBCL.pl')['model']
	# model.load_state_dict(model_dict)
	input = test_dataset[0].strip()
	pdb_id = PDBid[0]
	chain_id = chain[0]
	model.eval()
	logits = model.get_logits(input)
	logits = logits.view(-1, logits.size(-1))
	logits = logits[1:-1]
	# logits = torch.unsqueeze(logits, 0)
	pred_prob_all = F.softmax(logits, dim=1)
	pred_prob_sort = torch.max(pred_prob_all, 1)
	pred_class = pred_prob_sort[1]
	pred_class = pred_class.view(-1)
	pred_class = list(pred_class.cpu().numpy())

	r = requests.get("https://files.rcsb.org/view/" + pdb_id + ".pdb")
	path_data = '/data/result/proteinFunction/' + config.learn_name + '/pdb_file/'
	if not os.path.exists(path_data):
		os.mkdir(path_data)
	with open(os.path.join(path_data, pdb_id + ".pdb"), "wb") as f:
		f.write(r.content)
	index_aa_dic = get_index_aa_dic(pdb_id, chain_id, config)
	output = ''
	binding_index = []
	for i in index_aa_dic:
		output += index_aa_dic[i]

	tmp = []
	str_a = input
	str_b = output
	# 遍历：从最长的开始
	for i in range(len(str_a), 0, -1):
		for j in range(len(str_a) + 1 - i):
			sub = str_a[j:j + i]
			# 得到子串，判断其是否在str_b中
			if sub in str_b:
				tmp.append(sub)
		# 找到公共子串，跳出外层循环
		if tmp:
			break
	tmp = ''.join(tmp)
	index = output.find(tmp)
	index_input = input.find(tmp)
	# print("input_seq", input)
	# print("pdb_seq", output)
	for i, b in enumerate(index_aa_dic):
		if i >= index and i <= len(tmp) + index - 1 and pred_class[i - index + index_input] == 1 and input[i - index + index_input] == index_aa_dic[b]:
			binding_index.append(b)
	#根据pdb_id和chain_id给序列input结合上相应的index并求出结合的index
	return input, pred_class, binding_index, pdb_id, chain_id

    # for index, seq in enumerate(seqs):
    #     logits_data_all, motif_all = motif(model, seq, index)
    #     problist.extend(logits_data_all)
    #     attentionlist.extend(motif_all)