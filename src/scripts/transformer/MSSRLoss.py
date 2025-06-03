from torch import nn
import torch

class MSSRLoss(nn.Module):
    def __init__(self, epsilon=1e-8, leverage_penalty=0.0):
        super().__init__()
        self.epsilon = epsilon
        self.leverage_penalty = leverage_penalty

    def forward(self, predictions, targets):
        if predictions.shape != targets.shape:
            predictions = predictions.view(targets.shape)
        market_timing_returns = predictions * targets  # returns from market timing signals


        mean_returns = market_timing_returns.mean(dim=1)
        second_moment = (market_timing_returns ** 2).mean(dim=1) + self.epsilon
        denom = torch.sqrt(second_moment)

        sharpe_like = mean_returns / denom
        loss = 1 - sharpe_like.mean()

        if self.leverage_penalty > 0:
            loss += self.leverage_penalty * (predictions ** 2).mean()

        return loss
