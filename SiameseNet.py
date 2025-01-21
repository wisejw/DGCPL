import torch
import torch.nn as nn


class SiameseNet(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNet, self).__init__()
        self.fc_layer = nn.Linear(input_dim, 64)
        self.relu_layer = nn.ReLU()
        self.classificaton_layer = nn.Linear(64 * 4, 1)
        # self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x1, x2):
        c1 = self.relu_layer(self.fc_layer(x1))
        c2 = self.relu_layer(self.fc_layer(x2))
        diff = torch.sub(c1, c2)
        multiply = torch.mul(c1, c2)
        v = torch.cat((c1, c2, diff, multiply), 1)
        # pred_prob = self.sigmoid_layer(self.classificaton_layer(v))
        logit = self.classificaton_layer(v)
        return logit

