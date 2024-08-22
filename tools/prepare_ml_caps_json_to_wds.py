import json
from loguru import logger
import uuid
from torch.utils import data
import torchaudio
from typing import Optional, Union
import pandas as pd
import webdataset as wds
import uuid

from dataclasses import dataclass
import soundfile as sf
from fire import Fire
from tqdm import tqdm
import io
import torch
from pathlib import Path
import yaml
import json
import random


@dataclass
class JsonWDSConfig:
    outputshard: Path
    data: Union[str, Path]
    shuffle: bool = True
    target_sample_rate: int = 16000
    compress: bool = True
    config_file: str = ''
    maxcount: int = 10_000
    maxsize: float = 1e9

    def __post_init__(self):
        self.outputshard = Path(
            self.outputshard) / f"{Path(self.data).stem}_%06d.tar.gz"

    @classmethod
    def from_config(cls, config_file: Union[Path, str], **overwrite_kwargs):
        with open(config_file) as con_read:
            yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
        # values from config file are all possible params
        return cls(
            **dict(yaml_config, config_file=config_file, **overwrite_kwargs))


torch.set_num_threads(1)


def _write_flac(data, sr):
    buffer_flac = io.BytesIO()
    buffer_flac.name = f'file_{uuid.uuid4().hex}.flac'
    sf.write(buffer_flac, data.numpy(), sr)
    byte_buf = buffer_flac.getbuffer()
    return byte_buf


def read_data(data_json, num_caps=1):
    audio_path = data_json.pop('audio')
    duration_sample = data_json.pop('duration_sample')

    try:
        if duration_sample > 480000:  # the longest is 30s, 16k
            start = random.randint(0, duration_sample - 480000)
            wav, sr = torchaudio.load(
                audio_path, frame_offset=start, num_frames=480000)
        else:
            wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        captions = {}
        source_lang = 'eng_Latn'  # the default language
        if num_caps == 1:  # to support wavcaps dataset to pack which its caption number is 1 and it has only eng_Latn caption
            captions[data_json['source_lang']] = [data_json['caption']]
        else:
            for i in range(num_caps):
                source_lang = data_json[f'source_lang_{i+1}']
                if source_lang not in captions:
                    captions[source_lang] = [data_json[f'caption_{i+1}']]
                else:
                    captions[source_lang].append(data_json[f'caption_{i+1}'])

        return wav, sr, audio_path, captions
    except:
        return None


def run(*,
        config_path: Optional[Union[str, Path, JsonWDSConfig]] = None,
        **overwrite_kwargs):
    if config_path is None:
        logger.warning(
            "If you wanted to pass a config file, please use --config_path")
        config_path = JsonWDSConfig(**overwrite_kwargs)
    if not isinstance(config_path, JsonWDSConfig):
        config_path = JsonWDSConfig.from_config(config_path,
                                                **overwrite_kwargs)
    _run(config_path)


def _run(config: JsonWDSConfig):
    output_shard = Path(config.outputshard)
    output_shard.parent.mkdir(parents=True, exist_ok=True)

    sink = wds.ShardWriter(
        f'{output_shard}',
        encoder=False,
        compress=config.compress,
        maxcount=config.maxcount,
        maxsize=1e9,
    )
    with open(config.data) as fp:
        json_infos = json.load(fp)
    data = json_infos['data']
    num_caps = json_infos['num_captions_per_audio']
    if config.shuffle:
        random.shuffle(data)

    for item in tqdm(data, unit='file', total=len(data), leave=False):
        return_values = read_data(item, num_caps)
        if return_values is None:
            continue
        wav, sr, filepath, captions = return_values

        uid_segment = f"{Path(filepath).stem.replace('.','_').replace('-','_')}_{uuid.uuid4().hex}"
        if sr != config.target_sample_rate:
            wav = torchaudio.functional.resample(wav, sr,
                                                 config.target_sample_rate)
        flac_data = _write_flac(wav.squeeze(), config.target_sample_rate)
        sink.write({
            '__key__':
            f'{uid_segment}',
            'flac':
            flac_data,
            'json':
            json.dumps({'captions': captions},
                       ensure_ascii=False).encode('utf-8')
        })
    sink.close()


if __name__ == "__main__":
    Fire(run)
