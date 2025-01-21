import torch
import torch.nn as nn


class LossFunc(nn.Module):
    def __init__(self, device, T=0.5):
        super(LossFunc, self).__init__()
        # Initialize loss functions
        self.crossEntropy = nn.BCELoss()
        self.mse = nn.MSELoss()
        # Sigmoid activation function
        self.sig = nn.Sigmoid()
        # Temperature parameter for knowledge distillation
        self.T = T
        self.device = device

    def forward(self, logit_h, logit_n, logit_t, target):
        # Apply sigmoid to the logits to get predicted probabilities
        y_H = self.sig(logit_h)    # For CRHG logits
        y_N = self.sig(logit_n)    # For LBG logits
        y_T = self.sig(logit_t)    # For Teacher logits

        # Knowledge distillation (KD): compute soft target probabilities with temperature scaling
        p0_c = self.sig(logit_h / self.T)
        p0_t = self.sig(logit_n / self.T)
        p0_enm = self.sig(logit_t / self.T)
        # Compute the KD loss(L1 norm): sum of absolute differences between soft targets
        loss_kd = (torch.sum(torch.abs(p0_enm-p0_c)) + torch.sum(torch.abs(p0_enm-p0_t)))

        # Initialize loss and prediction
        loss = torch.Tensor([0.0]).cuda()
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)

        # Compute the BCE loss for all three predictions
        loss = loss + self.crossEntropy(y_H, target) + self.crossEntropy(y_N, target) + self.crossEntropy(y_T, target)
        # Average the three predictions for final output
        p_mean = (y_H + y_N + y_T) / 3.0

        # Concatenate predictions and ground truth for metrics calculation
        prediction = torch.cat([prediction, p_mean])
        ground_truth = torch.cat([ground_truth, target])

        return loss, loss_kd,  prediction, ground_truth

