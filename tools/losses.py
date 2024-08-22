#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioTextContrastiveLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,
                sim_a2t,
                sim_t2a,
                sim_targets=None):
        if sim_targets is None:
            sim_targets = torch.zeros(sim_a2t.size()).to(
                sim_a2t.device
            )
            sim_targets.fill_diagonal_(1)

        loss_a2t = - torch.sum(
            F.log_softmax(sim_a2t, dim=1) * sim_targets, dim=1
        ).mean()

        loss_t2a = - torch.sum(
            F.log_softmax(sim_t2a, dim=1) * sim_targets, dim=1
        ).mean()

        loss_atc = (loss_a2t + loss_t2a) / 2
        return loss_atc
