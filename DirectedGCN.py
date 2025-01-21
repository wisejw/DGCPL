import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout_rate=0.5):
        super(GCN, self).__init__()
        # Define graph convolution layers
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        # self.gc3 = GraphConvolution(nclass, nclass)

        # Define fully connected layers to map input features to hidden and output layers
        self.linear1 = nn.Linear(nfeat, nhid)
        self.linear2 = nn.Linear(nhid, nclass)
        # self.linear3 = nn.Linear(nclass, nclass)

        # Define dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adj):
        # First graph convolution + ReLU activation + Fully connected layer + Dropout
        x1 = self.dropout(F.relu(self.gc1(x, adj)) + self.linear1(x))
        # Second graph convolution + ReLU activation + Fully connected layer + Dropout
        x2 = self.dropout(F.relu(self.gc2(x1, adj)) + self.linear2(x1))
        # If a third graph convolution layer is needed, apply similarly
        # x3 = self.dropout(F.relu(self.gc2(x2, adj)) + self.linear2(x2))
        return x2
