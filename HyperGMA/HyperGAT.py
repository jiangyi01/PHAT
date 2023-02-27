import torch
from torch import nn
import torch.nn.functional as F
from layers import HyperGraphAttentionLayerSparse


class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.3):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.gat1_1 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False,
                                                     concat=True)
        self.gat1_2 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False,
                                                     concat=True)
        self.gat1_3 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False,
                                                     concat=True)
        self.gat1_4 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False,
                                                     concat=True)
        self.gat1_5 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False,
                                                     concat=True)
        self.gat1_6 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False,
                                                     concat=True)

        self.layer1_1 = nn.Linear(1024 * 6, 1024 * 12)
        self.layer1_2 = nn.Linear(1024 * 12, 1024 * 3)
        self.layer1_3 = nn.Linear(1024 * 3, 1024)

        self.gat1 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
                                                   concat=False)
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
                                                   concat=False)

        self.dropout = dropout



    def forward(self, x, H):
        x1_1 = self.gat1_1(x, H)
        x1_2 = self.gat1_2(x, H)
        x1_3 = self.gat1_3(x, H)
        x1_4 = self.gat1_4(x, H)
        x1_5 = self.gat1_5(x, H)
        x1_6 = self.gat1_6(x, H)

        x = torch.cat((x1_1, x1_2, x1_3, x1_4, x1_5, x1_6), dim=2)
        x = self.layer1_1(x)
        x = torch.relu(x)
        x = self.layer1_2(x)
        x = torch.relu(x)
        x = self.layer1_3(x)
        x = self.gat1(x, H)
        x = torch.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gat2(x, H)
        return x
