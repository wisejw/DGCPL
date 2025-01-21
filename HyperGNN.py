from torch import nn
import torch.nn.functional as F
from layers import HGNN_conv


class HGNN(nn.Module):
    def __init__(self, in_ch, n_hid, n_class, dropout_rate=0.5):
        super(HGNN, self).__init__()
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)
        # self.hgc3 = HGNN_conv(n_class, n_class)
        self.linear1 = nn.Linear(in_ch, n_hid)
        self.linear2 = nn.Linear(n_hid, n_class)
        # self.linear3 = nn.Linear(n_class, n_class)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, G):
        x1 = self.dropout(F.relu(self.hgc1(x, G)) + self.linear1(x))
        x2 = self.dropout(F.relu(self.hgc2(x1, G)) + self.linear2(x1))
        # x3 = self.dropout(F.relu(self.hgc2(x2, G)) + self.linear2(x2))
        return x2
