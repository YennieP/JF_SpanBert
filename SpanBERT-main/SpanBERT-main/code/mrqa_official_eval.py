# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""Official evaluation script for the MRQA Workshop Shared Task.
Adapted fromt the SQuAD v1.1 official evaluation script.
Usage:
    python official_eval.py dataset_file.jsonl.gz prediction_file.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pathlib import Path
from urllib.parse import urlparse
import argparse
import string
import re
import json
import gzip
import sys
import os
from collections import Counter

"""
mrqa_official_eval.py 的核心功能是帮助研究人员或开发者快速评估问答模型的表现。
"""


def cached_path(url_or_filename, cache_dir = None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.

    核心功能是通过 cached_path 函数处理可能是URL或本地路径的输入,判断其类型并返回合适的文件路径。如果输入是URL,则下载文件并缓存;如果是本地路径,则检查文件是否存在并返回路径。
    """
    if cache_dir is None:
        cache_dir = os.path.dirname(url_or_filename)
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    url_or_filename = os.path.expanduser(url_or_filename)
    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https', 's3'):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == '':
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    
    这个函数 normalize_answer 的目的是对输入字符串进行标准化处理,具体包括将文本转为小写、去除标点符号、删除冠词("a", "an", "the"),并且移除多余的空格,以简化和规范文本内容。这样的处理通常用于自然语言处理任务中,如文本匹配或比较。
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    # 按顺序调用上述函数,对输入文本进行标准化处理
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    对模型的输出进行评估:
    函数 f1_score 计算两个文本(预测文本 prediction 和真实文本 ground_truth)之间的 F1 分数,用于评估文本匹配的精确度和召回率。F1 分数是精确度(precision)和召回率(recall)的调和平均数,是一种常用的衡量分类模型性能的指标。
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """对模型的输出进行评估: 计算预测文本 prediction 与真实文本 ground_truth 之间的精确匹配分数"""
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """对模型的输出进行评估: 计算预测结果在多个真实答案 ground_truths 中的最佳匹配分数"""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def read_predictions(prediction_file):
    """对模型的输出进行评估: 从指定的 JSON 文件中读取预测结果"""
    with open(prediction_file) as f:
        predictions = json.load(f)
    return predictions


def read_answers(gold_file):
    """
    从压缩的 JSON 文件中读取标准答案,并将这些答案存储在一个字典中,方便后续的评估和比较
    """
    answers = {}
    with gzip.open(gold_file, 'rb') as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            if i == 0 and 'header' in example:
                continue
            for qa in example['qas']:
                answers[qa['qid']] = qa['answers']
    return answers


def evaluate(answers, predictions, skip_no_answer=False):
    """
    这个函数 evaluate 用于评估模型的预测结果 predictions 与标准答案 answers 之间的匹配程度。评估的指标包括精确匹配分数(Exact Match, EM)和F1 分数。函数还提供了一个可选参数 skip_no_answer,用于控制是否在遇到没有答案的问题时跳过评分。
    """

    # 初始化评估指标 f1 和 exact_match
    f1 = exact_match = total = 0

    # 对于每一个问题,根据它是否被回答进行加分(1/0), total 即总分
    for qid, ground_truths in answers.items():
        if qid not in predictions:
            if not skip_no_answer:
                message = 'Unanswered question %s will receive score 0.' % qid
                print(message)
                total += 1
            continue
        total += 1
        prediction = predictions[qid]

        # 对于每一个问题，分别计算不同评估指标下的分数
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)

    # 对所有问题计算总体的 f1 和 exact_match 分数
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


if __name__ == '__main__':
    """
    脚本的入口点,负责解析命令行参数,读取数据和预测结果,计算评估指标,并将结果输出为 JSON 格式。
    """

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='Evaluation for MRQA Workshop Shared Task')
    
    # 添加命令行参数
    parser.add_argument('dataset_file', type=str, help='Dataset File')
    parser.add_argument('prediction_file', type=str, help='Prediction File')
    parser.add_argument('--skip-no-answer', action='store_true')

    # 解析命令行参数
    # 调用 parse_args() 方法解析命令行参数，并将结果存储在 args 对象中。
    # 通过 args.dataset_file、args.prediction_file 和 args.skip_no_answer 可以访问用户传入的参数值。
    args = parser.parse_args()

    # 读取标准答案和预测结果
    answers = read_answers(cached_path(args.dataset_file))
    predictions = read_predictions(cached_path(args.prediction_file))

    # 评估预测结果
    metrics = evaluate(answers, predictions, args.skip_no_answer)

    print(json.dumps(metrics))
