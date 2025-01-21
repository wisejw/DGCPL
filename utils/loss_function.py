import torch
import torch.nn as nn


class LossFunc(nn.Module):
    def __init__(self, device, T=0.5):
        super(LossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.sig = nn.Sigmoid()
        self.T = T
        self.device = device

    def forward(self, logit_h, logit_n, logit_t, target):
        y_H = self.sig(logit_h)
        y_N = self.sig(logit_n)
        y_T = self.sig(logit_t)

        p0_c = self.sig(logit_h / self.T)
        p0_t = self.sig(logit_n / self.T)
        p0_enm = self.sig(logit_t / self.T)
        loss_kd = (torch.sum(torch.abs(p0_enm-p0_c)) + torch.sum(torch.abs(p0_enm-p0_t)))

        loss = torch.Tensor([0.0]).cuda()
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)

        loss = loss + self.crossEntropy(y_H, target) + self.crossEntropy(y_N, target) + self.crossEntropy(y_T, target)
        p_mean = (y_H + y_N + y_T) / 3.0

        prediction = torch.cat([prediction, p_mean])
        ground_truth = torch.cat([ground_truth, target])

        return loss, loss_kd,  prediction, ground_truth

