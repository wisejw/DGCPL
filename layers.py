import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))    # Weight parameter
        # Optional bias parameter
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()    # Initialize weights and bias

    def reset_parameters(self):
        # Initialization for weights and bias
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        # Forward pass: apply weight matrix and adjacency matrix G
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, dropout=0., bias=True):
        # Initialize the graph convolution layer
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))    # Weight parameter
        # Optional bias parameter
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()    # Initialize weights and bias

    def reset_parameters(self):
        # Initialize weights and bias
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # Forward pass: input features are transformed using the weight matrix
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        # String representation of the layer
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
