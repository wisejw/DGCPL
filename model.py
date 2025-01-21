import torch.nn as nn
import torch.nn.functional as F
from HyperGNN import HGNN
from DirectedGCN import GCN
from SiameseNet import SiameseNet


class CPL(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, G, adj_in, adj_out, feature_matrix, dropout_rate=0.5):
        super(CPL, self).__init__()
        self.dropout_rate = dropout_rate
        '''initial pretrain-model feature'''
        self.feature_matrix = feature_matrix

        '''generate two graphs'''
        self.G = G
        self.adj_out = adj_out
        self.adj_in = adj_in

        '''Concept-Resource HyperGraph: CRHG'''
        self.hgnn = HGNN(in_ch=in_channels, n_hid=out_channels1, n_class=out_channels2, dropout_rate=self.dropout_rate)

        '''Learning Behavior Graph: LBG'''
        # Two different GCN layers for processing out-degree and in-degree graphs
        self.gcn1 = GCN(nfeat=in_channels, nhid=out_channels1, nclass=int(out_channels2))
        self.gcn2 = GCN(nfeat=in_channels, nhid=out_channels1, nclass=int(out_channels2))
        self.w1 = nn.Linear(out_channels2, out_channels2)
        self.w2 = nn.Linear(out_channels2, out_channels2)

        '''Gate Knowledge Distillation: GKD'''
        self.w3 = nn.Linear(out_channels2, out_channels2)
        self.w4 = nn.Linear(out_channels2, out_channels2)

        '''SiameseNet'''
        self.siameseNet1 = SiameseNet(out_channels2)
        self.siameseNet2 = SiameseNet(out_channels2)
        self.siameseNet3 = SiameseNet(out_channels2)

        # Sigmoid activation for output probabilities
        self.sigmoid = nn.Sigmoid()

    def forward(self, c1, c2):
        '''Concept-Resource HyperGraph: CRHG'''
        X_H = self.hgnn(self.feature_matrix, self.G)

        '''Learning Behavior Graph: LBG'''
        X_out = self.gcn1(self.feature_matrix, self.adj_out)
        X_in = self.gcn2(self.feature_matrix, self.adj_in)
        # X_N = torch.cat([X_out, X_in], -1)
        X_N = F.relu(self.w1(X_out) + self.w2(X_in))

        '''Gate Knowledge Distillation: GKD'''
        theta = self.sigmoid(self.w3(X_H) + self.w4(X_N))
        out_h = theta * X_H
        out_n = (1 - theta) * X_N
        X_T = out_h + out_n

        '''SiameseNet'''
        logit_H = self.siameseNet1(X_H[c1], X_H[c2])
        logit_N = self.siameseNet2(X_N[c1], X_N[c2])
        logit_T = self.siameseNet3(X_T[c1], X_T[c2])

        return logit_H, logit_N, logit_T

