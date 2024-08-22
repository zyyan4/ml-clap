#!/usr/bin/env python3
import argparse
from typing import Dict

from loguru import logger
import yaml
from tqdm import tqdm
from typing import Dict
import torch
import warnings
from dataclasses import dataclass, field

from dataset import DatasetConfig, create_webdataset
from models.mlclap_model import MLCLAPConfig, MLCLAP
from tools.utils import (
    setup_seed,
    t2a, a2t, log_results,
)


warnings.filterwarnings("ignore")
torch.set_num_threads(1)

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
main_logger = logger.bind(indent=1)


@dataclass
class Config:
    dataset_config: DatasetConfig
    mlclap_config:  MLCLAPConfig
    optim_args: Dict = field(default_factory=lambda: dict(
        optimizer_name='adam', lr=5e-4, warmup_epoch=2, betas=[0.9, 0.999], eps=1e-8, momentum=0.9))
    training: Dict = field(default_factory=lambda: dict(
        max_epoch=20, epoch_length=540))
    seed: int = 20
    device: str = 'cuda'
    eval_num_caps: int = 5
    infer_path: str = ''


class Infer():
    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = self.config.device
        # setup model
        self.model = MLCLAP(config.mlclap_config)
        assert config.infer_path != '', 'Please config the infer model path'
        self.init_model(config.infer_path)
        self.model.to(self.device)

    def init_model(self, checkpoint):
        if 'http' in checkpoint:
            state_dict = torch.hub.load_state_dict_from_url(
                checkpoint, map_location=self.device)
        else:
            state_dict = torch.load(checkpoint, map_location=self.device)
        main_logger.info(f'Init model from clap {checkpoint}')
        strict = True  # if self.config.model_init_strict else False
        state_dict_update = {}
        for key in state_dict.keys():
            if self.model.state_dict()[key].shape == state_dict[key].shape:
                state_dict_update[key] = state_dict[key]
        self.model.load_state_dict(state_dict_update, strict=strict)
        del state_dict
        return

    @torch.no_grad()
    def validate(self, dataloader, num_caps=5):
        self.model.eval()
        audio_embeds_all, text_embeds_all = [], []
        for batch in tqdm(dataloader):
            (audio, _), text, _, source_lang = batch
            audio_embeds = self.model.encode_audio(audio=audio.to(self.device))
            text_embeds = self.model.encode_text(text, source_lang[0])

            if isinstance(audio_embeds, tuple):
                audio_embeds = audio_embeds[0]
            audio_embeds_all.append(audio_embeds.cpu())
            text_embeds_all.append(text_embeds.cpu())

        audio_embeds_all = torch.cat(audio_embeds_all, dim=0).numpy()
        text_embeds_all = torch.cat(text_embeds_all, dim=0).numpy()

        # evaluate text to audio retrieval
        r1, r5, r10, r50, medr, meanr, mAP10 = t2a(
            audio_embeds_all, text_embeds_all, num_caps=num_caps)

        # evaluate audio to text retrieval
        r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a = a2t(
            audio_embeds_all, text_embeds_all, num_caps=num_caps)

        return {"t2a": [r1, r5, r10, r50, medr, meanr, mAP10],
                "a2t": [r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a]}

    def inference(self, data_type='test_ac', num_caps=5):
        test_dataloader = create_webdataset(
            self.config.dataset_config, data_type=data_type)
        return self.validate(test_dataloader, num_caps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="settings/mlclap.yaml", type=str,
                        help="Setting files")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        dataset_config = DatasetConfig(**config['dataset'])
        del config['dataset']
        mlclap_config = MLCLAPConfig(**config['mlclap'])
        del config['mlclap']
        config = Config(dataset_config=dataset_config,
                        mlclap_config=mlclap_config, **config)
    infer = Infer(config)

    # setup seed
    seed = config.seed
    setup_seed(seed)

    if isinstance(dataset_config.data, Dict):
        for data_type in dataset_config.data.keys():
            main_logger.info(f'==== {data_type} ====')
            metrics = infer.inference(
                data_type=data_type, num_caps=config.eval_num_caps)
            log_results(
                metrics, f'{data_type} for the model {config.infer_path}', main_logger, test=True)


if __name__ == '__main__':
    main()
