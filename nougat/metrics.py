"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
from multiprocessing import Pool
import re
from pathlib import Path
from collections import defaultdict
from typing import List

import numpy as np

import nltk
from nltk import edit_distance
from tqdm import tqdm

import orjson

inline_reg = re.compile(r"\\\((.*?)(?<!\\)\\\)")
display_reg = re.compile(r"\\\[(.+?)(?<!\\)\\\]")
table_reg = re.compile(r"\\begin\{tabular\}(.+?)(?:\\end\{tabular\}|$)", re.S)


def compute_metrics(pred: str, gt: str, minlen=4):
    """
    Compute metrics for a single prediction.

    Args:
        pred: The predicted text.
        gt: The ground truth text.
        minlen: The minimum length of the text to compute metrics.

    Returns:
        A dictionary of metrics.
    """

    metrics = {}
    if len(pred) < minlen or len(gt) < minlen:
        return metrics
    
    edit_dist_raw = edit_distance(pred, gt)  # Raw edit distance
    metrics["edit_dist"] = edit_dist_raw / max(len(pred), len(gt))
    reference = gt.split()
    hypothesis = pred.split()
    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    # try:
    #     metrics["meteor"] = nltk.translate.meteor_score([reference], hypothesis)
    # except LookupError:
    # metrics["meteor"] = np.nan
    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["precision"] = nltk.scores.precision(reference, hypothesis)
    metrics["recall"] = nltk.scores.recall(reference, hypothesis)
    metrics["f_measure"] = nltk.scores.f_measure(reference, hypothesis)
    
    # ExpRate metric
    if pred == gt:
        metrics["exp_rate"] = 1  
    else:
        metrics["exp_rate"] = 0 
    # tolerance-based ExpRate metrics (≤1, ≤2, ≤3 errors)
    metrics["exp_rate_≤1"] = 1 if edit_dist_raw <= 1 else 0
    metrics["exp_rate_≤2"] = 1 if edit_dist_raw <= 2 else 0
    metrics["exp_rate_≤3"] = 1 if edit_dist_raw <= 3 else 0
    return metrics


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=Path, help="results file")
    parser.add_argument(
        "-N", dest="N", type=int, help="number of samples", default=None
    )
    args = parser.parse_args()
    d = orjson.loads(args.json.read_text(encoding="utf-8"))
    args.pred = d["predictions"]
    args.gt = d["ground_truths"]
    if args.N is not None:
        args.pred = args.pred[: args.N]
        args.gt = args.gt[: args.N]
    return args


def split_text(pages: List[str]):
    """
    Split a list of pages into text, inline math, display math, and table blocks.

    Args:
        pages: The pages to split.
    """
    text, math, table = [], [], []
    for page in pages:
        for i, reg in enumerate([inline_reg, display_reg, table_reg]):
            matches = "\n".join(reg.findall(page))
            if i == 2:
                table.append(matches)
            elif i == 1:
                math[-1] += matches
            else:
                math.append(matches)
            page = reg.sub("", page)
        text.append(page.strip())

    return text, math, table


def get_metrics(gt: List[str], pred: List[str], pool: bool = True):
    metrics = defaultdict(list)
    if pool:
        with Pool() as p:
            _metrics = p.starmap(compute_metrics, iterable=zip(pred, gt))
    else:
        _metrics = [compute_metrics(p, g) for p, g in zip(pred, gt)]
    for m in _metrics:
        for key, value in m.items():
            metrics[key].append(value)
    return dict(metrics)


if __name__ == "__main__":
    args = get_parser()
    for name, entries in zip(["gt", "pred"], [args.gt, args.pred]):
        full: Path = args.json.parent / (args.json.stem + "_" + name + "_full.mmd")
        full.write_text("\n\n------------------\n\n".join(entries))
    for i, (gt, pr) in enumerate(zip(split_text(args.gt), split_text(args.pred))):
        sub = ["Text", "Math", "Tables"][i]
        prpath: Path = args.json.parent / (
            args.json.stem + "_pred_" + sub.lower() + ".mmd"
        )
        prpath.write_text("\n\n------------------\n\n".join(pr))
        gtpath: Path = args.json.parent / (
            args.json.stem + "_gt_" + sub.lower() + ".mmd"
        )
        gtpath.write_text("\n\n------------------\n\n".join(gt))
        print("Results for", sub)

        metrics = get_metrics(gt, pr)
        print({key: sum(values) / len(values) for key, values in metrics.items()})
