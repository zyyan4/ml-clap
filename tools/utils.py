#!/usr/bin/env python3
import sys
import random
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from sentence_transformers import util


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_logger(exp_name, outputs):
    log_output_dir = Path(outputs, exp_name, 'logging')
    model_output_dir = Path(outputs, exp_name, 'models')
    log_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)
    logger.add(log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    return model_output_dir, log_output_dir


def setup_seed(seed):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_results(results, dataset, main_logger, test=False):
    if test:
        pre = "test"
    else:
        pre = "val"
    main_logger.info('{}: Caption to audio: r1: {:.2f}, r5: {:.2f}, '
                     'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}, mAP10: {:.3f}'.format(dataset, *results["t2a"]))
    # 'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}, mAP10: {:.3f}, ranks: {}, top1: {}'.format(dataset, *results["t2a"]))
    main_logger.info('{}: Audio to caption: r1: {:.2f}, r5: {:.2f}, '
                     'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}, mAP10: {:.3f}'.format(dataset, *results["a2t"]))


def remove_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def a2t(audio_embs, cap_embs, return_ranks=False, num_caps=5):
    # audio to caption retrieval
    num_audios = int(audio_embs.shape[0] / num_caps)

    ranks = np.zeros(num_audios)
    top1 = np.zeros(num_audios)
    AP10 = np.zeros(num_audios)
    for index in range(num_audios):
        # get query audio
        audio = audio_embs[num_caps * index]

        # compute scores
        # d = audio @ cap_embs.T
        d = util.cos_sim(torch.Tensor(audio), torch.Tensor(
            cap_embs)).squeeze(0).numpy()
        inds = np.argsort(d)[::-1]

        inds_map = []

        rank = 1e20
        for i in range(num_caps * index, num_caps * index + num_caps, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
            if tmp < 10:
                inds_map.append(tmp + 1)
        inds_map = np.sort(np.array(inds_map))
        # calculate average precision
        if len(inds_map) != 0:
            AP10[index] = np.sum(
                (np.arange(1, len(inds_map) + 1) / inds_map)) / num_caps
        else:
            AP10[index] = 0.
        ranks[index] = rank
        top1[index] = inds[0]

    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(AP10) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, mAP10, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr, mAP10


def t2a(audio_embs, cap_embs, return_ranks=False, num_caps=5):
    # caption to audio retrieval
    num_audios = int(audio_embs.shape[0] / num_caps)

    audios = np.array([audio_embs[i]
                      for i in range(0, audio_embs.shape[0], num_caps)])

    ranks = np.zeros(num_caps * num_audios)
    top1 = np.zeros(num_caps * num_audios)

    for index in range(num_audios):

        # get query captions
        queries = cap_embs[num_caps * index: num_caps * index + num_caps]

        # compute scores
        # d = queries @ audios.T
        d = util.cos_sim(torch.Tensor(queries), torch.Tensor(audios)).numpy()

        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[num_caps * index + i] = np.where(inds[i] == index)[0][0]
            top1[num_caps * index + i] = inds[i][0]

    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * \
        np.sum(1 / (ranks[np.where(ranks < 10)[0]] + 1)) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, mAP10, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr, mAP10
