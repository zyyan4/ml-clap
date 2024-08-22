#!/usr/bin/env python

from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from itertools import groupby

from models.audio_encoder import AudioEncoderWrapper
from models.text_encoder import TextEncoderSonarWrapper
from tools.losses import AudioTextContrastiveLoss


@dataclass
class MLCLAPConfig:
    audio_encoder: Dict = field(default_factory=lambda: dict(encoder_type='ced_base',
                                                             pretrained=True,
                                                             time_patch_out=0.0,
                                                             freq_patch_out=0.0,
                                                             ))
    embed_size: int = 1024
    temperature: float = 0.07
    embed_regularization: bool = True
    training: bool = True  # train() or eval()


class MLCLAP(nn.Module):
    def __init__(self, config: MLCLAPConfig, device: str = 'cuda') -> None:
        super().__init__()
        # audio encoder
        self.audio_encoder = AudioEncoderWrapper(model_type=config.audio_encoder['encoder_type'], pretrained=config.audio_encoder['pretrained'],
                                                 time_patch_out=config.audio_encoder['time_patch_out'], freq_patch_out=config.audio_encoder['freq_patch_out'], training=config.training, device=device)
        audio_width = self.audio_encoder.embed_dim
        # text encoder
        self.text_encoder = TextEncoderSonarWrapper(
            training=config.training, device=device)
        text_width = self.text_encoder.text_width
        # project
        embed_size = config.embed_size
        self.audio_proj = nn.Sequential(nn.Linear(
            audio_width, embed_size), nn.ReLU(), nn.Linear(embed_size, embed_size))
        self.text_proj = nn.Sequential(nn.Linear(
            text_width, embed_size), nn.ReLU(), nn.Linear(embed_size, embed_size))

        #
        self.temperature = nn.Parameter(torch.ones([]) * config.temperature)
        self.embed_reg = config.embed_regularization
        self.atc_loss = AudioTextContrastiveLoss()

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        audio_embeds = self.audio_encoder(audio)
        audio_embeds = F.normalize(self.audio_proj(audio_embeds), dim=-1)
        return audio_embeds

    def encode_text(self, text: List[str], source_lang: str = 'eng_Latn') -> torch.Tensor:
        text_embeds = self.text_encoder(text, source_lang)
        text_embeds = F.normalize(self.text_proj(text_embeds), dim=-1)
        return text_embeds

    def forward(self, audio: torch.Tensor, text: List[str], source_lang: List[str], idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        audio_embeds = self.encode_audio(audio)

        if len(set(source_lang)) > 1:  # include multilingual captions, need to split the batch
            group_source_langs = [list(group)
                                  for _, group in groupby(source_lang)]
            cur_idx = 0
            text_embeds = []
            for group_source_lang in group_source_langs:
                end_idx = cur_idx + len(group_source_lang)
                text_embeds.append(self.encode_text(
                    text[cur_idx: end_idx], group_source_lang[0]))
                cur_idx = end_idx
            text_embeds = torch.vstack(text_embeds)
        else:
            text_embeds = self.encode_text(text, source_lang[0])

        if idx is not None:
            idx = idx.view(-1, 1)
            pos_idx = torch.eq(idx, idx.t()).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        else:
            sim_targets = None

        sim_a2t = audio_embeds @ text_embeds.t() / self.temperature
        sim_t2a = text_embeds @ audio_embeds.t() / self.temperature
        loss = self.atc_loss(sim_a2t, sim_t2a, sim_targets)
        if self.embed_reg:
            loss = loss + torch.mean(torch.abs(audio_embeds)) / torch.sqrt(torch.sum(audio_embeds**2)) + \
                torch.mean(torch.abs(text_embeds)) / \
                torch.sqrt(torch.sum(text_embeds**2))

        return loss


if __name__ == '__main__':
    import sys
    import yaml
    with open(sys.argv[1], 'r') as fp:
        config = yaml.safe_load(fp)
    # update Parameter
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = MLCLAPConfig(**config)
    ml_clap = MLCLAP(config, device=device).to(device)
    import torch
    audio_data = torch.randn(1, 160000).to(device)
    text = ['It is a test text.']
    loss = ml_clap(audio_data, text, torch.tensor([1]).to(device))
    print(f'{loss=}')
