import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.utils.data
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import webdataset as wds
import random
from re import sub

random_state = np.random.RandomState(20)

bucket_boundaries = np.linspace(*(5, 30, 6))


@dataclass
class DatasetConfig:
    data: Union[Dict[str, str], List[str]]
    batch_size: int = 32
    eval_batch_size: int = 128
    max_length: float = 30
    min_length: float = 0.2
    max_token_size: int = 50
    sample_rate: int = 16000
    sampler: Optional[str] = None
    shuffle: Optional[int] = 256
    resample: bool = False
    normalize_amplitude: bool = False
    num_workers: int = 4
    mix_train: bool = False
    ml_percent: float = 0.1
    mix_langs: List[str] = field(default_factory=lambda: ['eng_Latn'])


def _batchsampler_by_length(data, bufsize=1000, initial=100, length_index: int = 1, batch_size: int = 128, sr: int = 16000):
    """Shuffle the data in the stream.

    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.

    data: iterator
    bufsize: buffer size for shuffling
    length_index: int The index where to find the length to sort
    returns: iterator
    rng: either random module or random.Random instance

    """
    initial = min(initial, bufsize)
    buf = []
    lengths = []
    ind_n_len = []

    def __element_to_bucket_id(seq_length):
        boundaries = list(bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
            np.less_equal(buckets_min, seq_length),
            np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id

    def __mk_buckets(lengths):
        for i, length in enumerate(lengths):
            ind_n_len.append((i, length))
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number.
        for p, seq_len in ind_n_len:
            pid = __element_to_bucket_id(seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():
            # random_state.shuffle(data_buckets[k])
            data_buckets[k] = torch.tensor(data_buckets[k])
        return data_buckets

    def __return_buf_batch(buf, lengths):
        buf_batches = []
        data_buckets = __mk_buckets(lengths)
        for k in data_buckets.keys():
            # if k < 3 else max(batch_size // 8, 1))
            index_splits = data_buckets[k].split(batch_size)
            for index_split in index_splits:
                buf_batch = [buf[index] for index in index_split]
                # x[-1] represent source_lang
                buf_batch = sorted(buf_batch, key=lambda x: x[-1])
                buf_batches.append(buf_batch)
        return buf_batches

    for sample in data:
        buf.append(sample)
        lengths.append(float(sample[length_index].shape[-1] / sr))
        if len(buf) < bufsize:
            try:
                next_data = next(data)
                buf.append(next_data)  # skipcq: PYL-R1708
                lengths.append(float(next_data[length_index].shape[-1] / sr))
            except StopIteration:
                pass
        if len(buf) >= initial:
            buf_batches = __return_buf_batch(buf, lengths)
            for buf_batch in buf_batches:
                yield buf_batch
            buf = []
            lengths = []
            ind_n_len = []
    while len(buf) > 0:
        buf_batches = __return_buf_batch(buf, lengths)
        for buf_batch in buf_batches:
            yield buf_batch
        buf = []
        lengths = []
        ind_n_len = []


batchsampler_by_length = wds.pipelinefilter(_batchsampler_by_length)


def text_preprocess(sentence):

    # transform to lower case
    sentence = sentence.lower()

    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([，。！？：;、,.!?;:"](?:\s|$))',
                   r'\1', sentence).replace('  ', ' ')

    # remove punctuations
    sentence = sub('[(，。！？;：、,.!?;:|*\")]', ' ', sentence).replace('  ', ' ')
    return sentence


def _audio_to_mono(audio):
    if isinstance(audio, tuple):
        audio = audio[0]
    if audio.ndim == 2:
        audio = audio.mean(0)
    return audio


def _augment(data,
             max_size: int,
             min_size: int,
             max_token_size: int,
             drop_clipped: bool = True,
             normalize_amplitude: bool = False,
             mix_train: bool = False,
             ml_percent: float = 0.1,
             mix_langs: List[str] = ['eng_Latn'],
             ):
    def __caption_len(caption, source_lang):
        if source_lang in ['zho_Hans', 'jpn_Jpan']:
            return len(caption.replace(' ', '').encode('utf-8')) // 3
        else:
            return len(caption.split())

    for sample in data:
        # ml_captions is a dict, like this {'eng_Latn': ['xxxx', 'xxdddd'], 'fra_Latn': ['oododo', 'afjshdfj']}
        audio, ml_captions, uid = sample
        if audio.abs().max() >= 0.99 and drop_clipped:
            continue
        if normalize_amplitude:
            max_energy = torch.max(torch.abs(audio), dim=-1)[0]
            scale = (1. / max_energy).clamp(0.1, 10)
            audio = audio * scale
        audio_crop = audio
        if isinstance(ml_captions, Dict):
            # First, get all eng_Latn caption
            assert 'eng_Latn' in ml_captions, 'eng_Latn must be in captions for multilingual clap'
            captions_pickup = [
                {'eng_Latn': caption} for caption in ml_captions['eng_Latn']]
            ml_captions.pop('eng_Latn')
            # Second, select a specified percent of multilingual captions randomly
            if mix_train:
                for source_lang, captions in ml_captions.items():
                    assert source_lang in ['fra_Latn', 'deu_Latn', 'spa_Latn', 'nld_Latn',
                                           'cat_Latn', 'jpn_Jpan', 'zho_Hans'], 'No support languages !!!!'
                    if source_lang not in mix_langs:
                        continue
                    assert 'eng_Latn' != source_lang, 'source_lang error'
                    for caption in captions:
                        if random.random() <= ml_percent and __caption_len(caption, source_lang) <= max_token_size:
                            captions_pickup.append({source_lang: caption})
            for caption_ele in captions_pickup:
                for source_lang, caption in caption_ele.items():
                    caption = text_preprocess(caption)
                    if audio.shape[-1] > max_size:
                        start = random.randint(0, audio.shape[-1] - max_size)
                        audio_crop = audio[start: start + max_size]
                    yield (audio_crop, caption, uid, source_lang)
        else:
            captions_pickup = ml_captions

            for caption in captions_pickup:
                caption = text_preprocess(caption)
                if audio.shape[-1] > max_size:
                    start = random.randint(0, audio.shape[-1] - max_size)
                    audio_crop = audio[start: start + max_size]
                yield (audio_crop, caption, uid, 'eng_Latn')


class Audiowebdataset_Fluid_MLCLAP(wds.DataPipeline):

    def __init__(
        self,
        urls,
        shuffle: Optional[int] = None,
        resample: bool = False,
        sample_rate: int = 16000,
        crop_shuffle: Optional[int] = None,
        max_size: int = 480000,  # 30s
        min_size: int = 32000,  # 2s
        max_token_size: int = 50,
        batch_size: int = 1,
        drop_clipped: bool = False,
        normalize_amplitude: bool = False,
        mix_train: bool = False,
        ml_percent: float = 0.1,
        mix_langs: List[str] = ['eng_Latn'],
    ):
        from functools import partial

        self.urls = urls
        pipeline: List = [
            wds.SimpleShardList(urls)
            if resample is False else wds.ResampledShards(urls)
        ]
        if shuffle is not None:
            # Tar wise shuffle
            pipeline.extend([
                wds.detshuffle(
                    bufsize=shuffle,
                    initial=shuffle // 4,
                ),
                wds.split_by_node,
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker at each node
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.shuffle(
                    bufsize=shuffle,
                    initial=shuffle // 4,
                ),
            ])
        else:
            pipeline.extend([wds.split_by_worker, wds.tarfile_to_samples()])

        def filter_samplingrate(data_stream):
            for sample in data_stream:
                audio, *extra = sample
                if isinstance(audio, tuple) and len(audio) == 2:
                    if audio[-1] != sample_rate:
                        continue
                yield sample

        pipeline.extend([
            wds.decode(wds.torch_audio, handler=wds.warn_and_continue),
            wds.to_tuple("mp3;wav;flac", 'json', '__key__'),
            filter_samplingrate,
            # x['cation']是一个list, 用于支持多个caption的情况
            wds.map_tuple(partial(_audio_to_mono),
                          lambda x: x['captions'], lambda x: x),
            partial(_augment,
                    max_size=max_size,
                    min_size=min_size,
                    max_token_size=max_token_size,
                    drop_clipped=drop_clipped,
                    normalize_amplitude=normalize_amplitude,
                    mix_train=mix_train,
                    ml_percent=ml_percent,
                    mix_langs=mix_langs,
                    )
        ])
        if crop_shuffle is not None:
            pipeline.append(wds.shuffle(crop_shuffle))

        # if batch_size is not None:
            # Batch but do not merge into tensors yet
        pipeline.append(
            wds.batched(batch_size,
                        collation_fn=partial(
                            wds.filters.default_collation_fn,
                            combine_tensors=False)))
        super().__init__(pipeline)


def expand_with_brace(lists: List[str]):
    import braceexpand
    r = []
    for l in lists:
        if '*' in l:
            # Expand using "posix" based *
            l = braceexpand.braceexpand(l)
            for expand_l in l:
                r.extend(map(str, Path(expand_l).parent.glob(Path(expand_l).name)))
        else:
            r.extend(braceexpand.braceexpand(l))
    return r


def pad(tensorlist: Sequence[torch.Tensor], padding_value: float = 0.):
    # Tensors are expected to be B, ..., T
    lengths = [f.shape[-1] for f in tensorlist]
    dims = tensorlist[0].shape
    trailing_dims = dims[:-1]
    batch_dim = len(lengths)
    num_raw_samples = max(lengths)
    out_dims = (batch_dim, ) + trailing_dims + (num_raw_samples, )
    out_tensor = torch.full(out_dims,
                            fill_value=padding_value,
                            dtype=tensorlist[0].dtype)
    for i, tensor in enumerate(tensorlist):
        length = tensor.shape[-1]
        out_tensor[i, ..., :length] = tensor[..., :length]
    return out_tensor, torch.as_tensor(lengths)


def collate_with_lengths_wds(samples, combine_scalars=True, combine_tensors=True, sample_sort=True):
    samples = samples[0] if sample_sort else samples
    batched = list(zip(*samples))
    result = []
    for idx, b in enumerate(batched):
        if isinstance(b[0], (int, float)):
            if combine_scalars:
                b = np.array(list(b))
        elif isinstance(b[0], torch.Tensor):
            if combine_tensors:
                b = pad(list(b))
        elif isinstance(b[0], np.ndarray):
            if combine_tensors:
                b = np.array(list(b))
        else:
            b = list(b)
        result.append(b)
    return result


def create_webdataset(config_parameters: DatasetConfig, data_type: str = 'train'):
    dataset_kwargs = dict(
        shuffle=config_parameters.shuffle if data_type == 'train' else None,
        sample_rate=config_parameters.sample_rate,
        max_size=int(config_parameters.max_length *
                     config_parameters.sample_rate),
        min_size=int(config_parameters.min_length *
                     config_parameters.sample_rate),
        max_token_size=int(config_parameters.max_token_size),
        resample=config_parameters.resample if data_type == 'train' else False,
        batch_size=config_parameters.batch_size,
        normalize_amplitude=config_parameters.normalize_amplitude,
        mix_train=config_parameters.mix_train,
        ml_percent=config_parameters.ml_percent,
        mix_langs=config_parameters.mix_langs,
    )
    print(dataset_kwargs)

    data_path: List[str] = config_parameters.data if isinstance(
        config_parameters.data, List) else config_parameters.data[data_type]
    dataset = Audiowebdataset_Fluid_MLCLAP(
        expand_with_brace(data_path), **dataset_kwargs)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=config_parameters.num_workers if data_type == 'train' else 1,
    ).unbatched()
    if data_type == 'train':
        dataloader = dataloader.compose(batchsampler_by_length(initial=2560, bufsize=2560, length_index=0, batch_size=config_parameters.batch_size,
                                        sr=config_parameters.sample_rate)).batched(1, collation_fn=lambda data: collate_with_lengths_wds(data))
    else:
        dataloader = dataloader.batched(
            config_parameters.batch_size, collation_fn=lambda data: collate_with_lengths_wds(data, sample_sort=False))

    return dataloader


if __name__ == '__main__':
    import sys
    import yaml

    with open(sys.argv[1]) as fp:
        config_parameters = yaml.safe_load(fp)['dataset']
    config_parameters = DatasetConfig(**config_parameters)
    data_loader = create_webdataset(
        config_parameters, data_type='train')  # .with_length(532).with_epoch(532)

    import time
    t1 = time.time()
    print(f'start {t1}')
    for idx, data in enumerate(data_loader):
        print(idx)
    print(f'time consume: {time.time() - t1}')
