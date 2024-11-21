#!/usr/bin/env python3
import torch
import torch.nn as nn

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from fairseq2.data import Collater
from fairseq2.data.data_pipeline import read_sequence
from fairseq2.data.text import read_text

from sonar.inference_pipelines.text import extract_sequence_batch
from sonar.models.sonar_text import (
    load_sonar_text_encoder_model,
    load_sonar_tokenizer,
)


class TextEncoderSonarWrapper(nn.Module):
    def __init__(self, tokenizer: str = "text_sonar_basic_encoder", encoder: str = "text_sonar_basic_encoder", max_seq_len: Optional[int] = None, training: bool = True):
        super().__init__()
        self.tokenizer = load_sonar_tokenizer(tokenizer, progress=True)
        self.model = load_sonar_text_encoder_model(
            encoder, progress=True)
        self.model = self.model.train() if training else self.model.eval()

        self.max_seq_len = self.model.encoder_frontend.pos_encoder.max_seq_len if max_seq_len is None else max_seq_len
        self.text_width = 1024
        self.training = self.model.training

    def forward(self, input: Union[Path, Sequence[str]], source_lang: str = "eng_Latn") -> torch.Tensor:
        tokenizer_encoder = self.tokenizer.create_encoder(lang=source_lang)

        n_truncated = 0

        def truncate(x: torch.Tensor) -> torch.Tensor:
            if x.shape[0] > self.max_seq_len:
                nonlocal n_truncated
                n_truncated += 1
            return x[:self.max_seq_len]

        pipeline: Iterable = (
            (
                read_text(input)
                if isinstance(input, (str, Path))
                else read_sequence(input)
            )
            .map(tokenizer_encoder)
            .map(truncate)
            .bucket(128)
            .map(Collater(self.tokenizer.vocab_info.pad_idx))
            .map(lambda x: extract_sequence_batch(x, self.device))
            .map(self.model)
            .map(lambda x: x.sentence_embeddings.to(self.device))
            .and_return()
        )
        results: List[torch.Tensor] = list(iter(pipeline))
        sentence_embeddings = torch.cat(results, dim=0)
        return sentence_embeddings


if __name__ == '__main__':
    sentences = ['My name is SONAR.', 'I can embed the sentences into vectorial space.',
                 'ok', 'yes', 'test', 'me and you', 'we need to go']
    model = TextEncoderSonarWrapper(device='cuda', training=False)
    #
    # total_params = sum(p.numel() for p in model.parameters())
    print("start")
    with torch.no_grad():
        text_output = model(sentences, source_lang='eng_Latn')
        print(text_output)
