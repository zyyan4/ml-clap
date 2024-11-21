#!/usr/bin/env python3

import torch
import torch.nn as nn
from models import ced


class AudioEncoderWrapper(nn.Module):
    def __init__(self, model_type: str = 'ced_base', training: bool = True, pretrained: bool = False, time_patch_out: float = 0.0, freq_patch_out: float = 0.0, **kwargs):
        super().__init__()
        self.model = getattr(ced, model_type)(pretrained=pretrained, pooling='cat',
                                              time_patch_out=time_patch_out, freq_patch_out=freq_patch_out, **kwargs)
        self.model = self.model.train() if training else self.model.eval()
        self.embed_dim = self.model.embed_dim
        self.training = self.model.training

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)


if __name__ == '__main__':
    # from models import ced
    # models = ced.list_models()
    model = AudioEncoderWrapper(
        model_type='ced_base', pretrained=True).to('cuda')
    x = torch.randn(2, 160000).to('cuda')
    y = model(x)
    print(y)
