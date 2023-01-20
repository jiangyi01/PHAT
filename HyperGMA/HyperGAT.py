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
        # self.gat1_7 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False,
        #                                              concat=True)
        # self.gat1_8 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False,
        #                                              concat=True)
        # self.gat1_9 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False,
        #                                              concat=True)
        # self.gat1_10 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_11 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_12 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_13 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_14 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_15 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_16 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_17 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_18 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_19 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_20 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_21 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_22 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_23 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_24 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_25 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_26 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_27 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_28 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_29 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_30 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_31 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_32 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_33 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_34 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_35 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_36 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_37 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_38 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_39 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)
        # self.gat1_40 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
        #                                               transfer=False,
        #                                               concat=True)

        self.layer1_1 = nn.Linear(1024 * 6, 1024 * 12)
        self.layer1_2 = nn.Linear(1024 * 12, 1024 * 3)
        self.layer1_3 = nn.Linear(1024 * 3, 1024)

        self.gat1 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
                                                   concat=False)
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
                                                   concat=False)

        self.dropout = dropout

        # self.gat2_1 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
        #                                              concat=False)
        # self.gat2_2 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
        #                                              concat=False)
        # self.gat2_3 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
        #                                              concat=False)
        # self.gat2_4 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
        #                                              concat=False)
        # self.gat2_5 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
        #                                              concat=False)
        # self.gat2_6 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
        #                                              concat=False)
        # self.gat2_7 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
        #                                              concat=False)
        # self.gat2_8 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
        #                                              concat=False)
        # self.gat2_9 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
        #                                              concat=False)
        # #
        # self.layer2_1 = nn.Linear(1024 * 9, 1024 * 6)
        # self.layer2_2 = nn.Linear(1024 * 6, 1024* 3)
        #
        # self.layer2_3 = nn.Linear(1024 * 3, 1024)

    def forward(self, x, H):
        x1_1 = self.gat1_1(x, H)
        x1_2 = self.gat1_2(x, H)
        x1_3 = self.gat1_3(x, H)
        x1_4 = self.gat1_4(x, H)
        x1_5 = self.gat1_5(x, H)
        x1_6 = self.gat1_6(x, H)
        # x1_7 = self.gat1_7(x, H)
        # x1_8 = self.gat1_8(x, H)
        # x1_9 = self.gat1_9(x, H)
        # x1_10 = self.gat1_10(x, H)
        # x1_11 = self.gat1_11(x, H)
        # x1_12 = self.gat1_12(x, H)
        # x1_13 = self.gat1_13(x, H)
        # x1_14 = self.gat1_14(x, H)
        # x1_15 = self.gat1_15(x, H)
        # x1_16 = self.gat1_16(x, H)
        # x1_17 = self.gat1_17(x, H)
        # x1_18 = self.gat1_18(x, H)
        # x1_19 = self.gat1_19(x, H)
        # x1_20 = self.gat1_20(x, H)
        # x1_21 = self.gat1_21(x, H)
        # x1_22 = self.gat1_22(x, H)
        # x1_23 = self.gat1_23(x, H)
        # x1_24 = self.gat1_24(x, H)
        # x1_25 = self.gat1_25(x, H)
        # x1_26 = self.gat1_26(x, H)
        # x1_27 = self.gat1_27(x, H)
        # x1_28 = self.gat1_28(x, H)
        # x1_29 = self.gat1_29(x, H)
        # x1_30 = self.gat1_30(x, H)
        # x1_31 = self.gat1_31(x, H)
        # x1_32 = self.gat1_32(x, H)
        # x1_33 = self.gat1_33(x, H)
        # x1_34 = self.gat1_34(x, H)
        # x1_35 = self.gat1_35(x, H)
        # x1_36 = self.gat1_36(x, H)
        # x1_37 = self.gat1_37(x, H)
        # x1_38 = self.gat1_38(x, H)
        # x1_39 = self.gat1_39(x, H)
        # x1_40 = self.gat1_40(x, H)
        # x = torch.cat((x1_1, x1_2, x1_3, x1_4, x1_5, x1_6, x1_7, x1_8, x1_9, x1_10, x1_11, x1_12, x1_13, x1_14, x1_15,
        #                x1_16, x1_17, x1_18,x1_19, x1_20), dim=2)
        x = torch.cat((x1_1, x1_2, x1_3, x1_4, x1_5, x1_6), dim=2)
        x = self.layer1_1(x)
        x = torch.relu(x)
        x = self.layer1_2(x)
        x = torch.relu(x)
        x = self.layer1_3(x)
        x = self.gat1(x, H)
        x = torch.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # x2_1 = self.gat2_1(x, H)
        # x2_2 = self.gat2_2(x, H)
        # x2_3 = self.gat2_3(x, H)
        # x2_4 = self.gat2_4(x, H)
        # x2_5 = self.gat2_5(x, H)
        # x2_6 = self.gat2_6(x, H)
        # x2_7 = self.gat2_7(x, H)
        # x2_8 = self.gat2_8(x, H)
        # x2_9 = self.gat2_9(x, H)
        # x = torch.cat((x2_1, x2_2, x2_3, x2_4, x2_5, x2_6, x2_7, x2_8,x2_9), dim=2)
        # x = self.layer2_1(x)
        # x = torch.relu(x)
        # x = self.layer2_2(x)
        # x = torch.relu(x)
        # x = self.layer2_3(x)
        x = self.gat2(x, H)
        return x
