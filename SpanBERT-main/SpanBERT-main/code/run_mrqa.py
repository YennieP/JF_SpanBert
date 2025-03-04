# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""Run BERT on MRQA."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import time
import sys
import gzip
from io import open

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
from mrqa_official_eval import exact_match_score, f1_score, metric_max_over_ground_truths

"""定义文件路径"""
PRED_FILE = "predictions.json"
EVAL_FILE = "eval_results.txt"
TEST_FILE = "test_results.txt"

"""配置日志记录"""
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class MRQAExample(object):
    """
    A single training/test example for the MRQA dataset.
    For examples without an answer, the start and end position are -1.

    MRQAExample类是用来表示机器阅读理解问答(Machine Reading Comprehension for Question Answering, MRQA)数据集中的单个训练或测试示例的。
    
    这个类定义了一个问答示例的基本结构,包括问题ID、问题文本、文档分词(即文档被分割成的一系列词或标记)、原始答案文本(如果有的话),以及答案在文档分词中的起始和结束位置(这两个位置用于确定答案的具体范围)

    它提供了一种结构化的方式来存储和访问问答示例的信息
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class InputFeatures(object):
    """
    A single set of features of data.
    
    InputFeatures 类是用于表示单个数据特征集合的类,这些数据特征通常用于机器学习或深度学习模型的输入。在机器阅读理解(Machine Reading Comprehension, MRC)或类似的任务中,这个类可能用于封装从原始文本数据(如问题和文档)转换而来的、模型可以直接处理的特征。
    """

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens # 标记列表
        self.token_to_orig_map = token_to_orig_map # 标记到原始索引的映射
        self.token_is_max_context = token_is_max_context # 标记是否是最大上下文,多个文档片段中,某些标记可能比其他标记包含更多上下文信息
        self.input_ids = input_ids # 将标记转换为模型可识别的ID
        self.input_mask = input_mask # 指示哪些位置是真实标记,哪些位置是填充标记
        self.segment_ids = segment_ids # 用于区分查询和文档标记,通常在模型中使用来区分两种不同类型的输入
        self.start_position = start_position
        self.end_position = end_position


def read_mrqa_examples(input_file, is_training):
    """
    Read a MRQA json file into a list of MRQAExample.
    
    这段代码定义了一个函数 read_mrqa_examples,它的目的是从一个MRQA格式的JSON文件中读取数据,并将这些数据转换为MRQAExample对象的列表。这个函数特别用于处理训练或测试数据,具体取决于is_training参数的值。
    """
    # 打开并读取文件
    with gzip.GzipFile(input_file, 'r') as reader:
        # skip header
        content = reader.read().decode('utf-8').strip().split('\n')[1:]
        input_data = [json.loads(line) for line in content]

    # 辅助函数: 用于检查给定的字符是否是空白字符
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    # 处理数据
    examples = []
    num_answers = 0
    for i, entry in enumerate(input_data):
        if i % 1000 == 0:
            logger.info("Processing %d / %d.." % (i, len(input_data)))
        paragraph_text = entry["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # 对于每个条目,首先处理文档文本(paragraph_text),将其分词为doc_tokens列表,并构建char_to_word_offset列表,该列表将文档中的每个字符位置映射到其对应的分词索引。
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        # 对于每个问答对,提取问题ID(qid)、问题文本(question)和(可选的)答案信息。
        for qa in entry["qas"]:
            qas_id = qa["qid"]
            question_text = qa["question"]
            start_position = None
            end_position = None
            orig_answer_text = None

            # 如果是在训练模式下(is_training=True),则解析答案信息(detected_answers),找到第一个答案的字符跨度(char_spans),并据此计算答案在分词列表中的起始和结束位置。同时,计算并更新答案的总数(num_answers)。
            if is_training:
                answers = qa["detected_answers"]
                # import ipdb
                # ipdb.set_trace()
                spans = sorted([span for spans in answers for span in spans['char_spans']])
                # take first span
                char_start, char_end = spans[0][0], spans[0][1]
                orig_answer_text = paragraph_text[char_start:char_end+1]
                start_position, end_position = char_to_word_offset[char_start], char_to_word_offset[char_end]
                num_answers += sum([len(spans['char_spans']) for spans in answers])

            example = MRQAExample(
                qas_id=qas_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position, # 表示答案在输入序列中的位置。
                end_position=end_position)
            examples.append(example)
    
    # 日志记录
    logger.info('Num avg answers: {}'.format(num_answers / len(examples)))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """
    Loads a data file into a list of `InputBatch`s.
    
    它的主要作用是将原始数据(例如,问题和文档对)转换为模型训练或推理所需的特征格式。

    convert_examples_to_features：这个函数接收原始数据(examples),一个分词器(tokenizer),以及一些配置参数(如最大序列长度max_seq_length、文档步长doc_stride、最大查询长度max_query_length和是否训练is_training),然后将这些数据转换为一系列InputBatch对象(这里实际上是InputFeatures对象列表)
    """

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        # 查询分词：将查询(问题)文本通过分词器转换为标记列表,并裁剪到最大长度
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # 文档分词: 将文档文本通过分词器转换为子标记列表,并构建原始索引到标记索引的映射。
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # 处理起始和结束位置: 在训练模式下,根据原始答案的起始和结束位置,转换为标记索引的起始和结束位置,并可能进行优化。
        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = -1
            tok_end_position = -1
        if is_training:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        # 滑动窗口处理长文档: 如果文档过长,无法一次性放入模型中,则使用滑动窗口方法将其分割成多个子序列。
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {} # 标记到原始索引的映射
            token_is_max_context = {} # 标记是否是最大上下文,多个文档片段中,某些标记可能比其他标记包含更多上下文信息
            segment_ids = [] # 用于区分查询和文档标记,通常在模型中使用来区分两种不同类型的输入
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens) # 将标记转换为模型可识别的ID

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids) # 指示哪些位置是真实标记,哪些位置是填充标记

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            # 表示答案在输入序列中的位置。
            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            if example_index < 5:
                # 记录日志
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """
    Returns tokenized answer spans that better match the annotated answer.

    _improve_answer_span 的目的是在给定的文档分词(doc_tokens)中,根据一个初始的答案起始位置(input_start)和结束位置(input_end),以及原始答案文本(orig_answer_text)和一个分词器(tokenizer),找到一个更好的、与原始答案文本完全匹配的答案区间。
    
    这个过程主要是通过调整起始和结束位置来实现的,以期望找到一个与orig_answer_text经过分词器处理后的文本完全相同的文本片段。
    """

    # 处理原始答案文本
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text)) # 分词并连接成一个空格分隔的字符串tok_answer_text

    # 遍历可能的起始和结束位置: 为了在原始答案位置的周围寻找一个更好的匹配。
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            # 提取并比较文本片段
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)]) # 从文档分词中提取一个文本片段,并连接成一个字符串text_span
            if text_span == tok_answer_text: # 进行比较
                return (new_start, new_end) # 找到匹配并返回

    return (input_start, input_end) # 如果遍历完所有可能的位置后都没有找到匹配的文本片段,则返回原始位置


def _check_is_max_context(doc_spans, cur_span_index, position):
    """
    Check if this is the 'max context' doc span for the token.
    
    _check_is_max_context 的目的是确定给定的文档片段（doc_spans 中的一个，通过索引 cur_span_index 指定）是否为某个位置（position）的"最大上下文"文档片段。
    
    在文本处理或自然语言处理的某些任务中，尤其是问答系统或阅读理解任务中，找到包含问题中提及的实体或短语的最大上下文片段是很重要的，因为这有助于更准确地理解上下文并提供更精确的答案。
    """

    # 这两个变量将用于跟踪得分最高的文档片段及其索引。
    best_score = None
    best_span_index = None

    # 遍历文档片段
    for (span_index, doc_span) in enumerate(doc_spans):
        # 检查给定的 position 是否在文档片段内,即起始位置和结束位置中间
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        
        # 计算左右上下文和得分
        num_left_context = position - doc_span.start # 该位置左侧的上下文数量
        num_right_context = end - position # 该位置右侧的上下文数量
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length # 计算得分: 取左右上下文数量的最小值，并加上一个小权重 -- 优先考虑包含更多上下文信息的文档片段，同时稍微偏向于更长的片段（尽管权重很小）。
        # 更新最佳文档片段
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    # 比较传入的 cur_span_index（当前考虑的文档片段的索引）和计算出的 best_span_index（得分最高的文档片段的索引）来确定当前文档片段是否为最大上下文文档片段
    return cur_span_index == best_span_index

# 名为 RawResult 的 namedtuple，它是 Python 中 collections 模块提供的一个工厂函数，用于创建一个简单的类，这个类可以用来存储和管理一组相关的数据。namedtuple 主要用于创建一个可以通过属性名访问元素内容的元组子类。
# 在这个例子中，RawResult 被设计用来存储和处理与某个“原始结果”相关的三个主要信息：unique_id、start_logits 和 end_logits
RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def make_predictions(all_examples, all_features, all_results, n_best_size,
                     max_answer_length, do_lower_case, verbose_logging):
    """
    处理从深度学习模型（如BERT等）中获取的问答系统预测结果，并将其转换为更易于理解和使用的格式。具体来说，它接收一系列输入（如所有示例、特征、结果等），并输出每个问题的预测答案以及一系列最佳候选答案的详细信息。
    """
    # 初始化数据结构
    example_index_to_features = collections.defaultdict(list) # example_index_to_features 的作用是将每个 example_index 映射到与其相关的特征列表上。
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    unique_id_to_result = {} # 映射每个特征的唯一ID到其对应的预测结果
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]) # 存储初步预测信息

    # 遍历所有示例
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples): # 对每个example
        features = example_index_to_features[example_index]
        prelim_predictions = []
        for (feature_index, feature) in enumerate(features): # 对每个特征列表
            result = unique_id_to_result[feature.unique_id] # 获取每个特征的预测结果
            # 根据起始和结束对数的最高得分生成初步预测列表
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True) # 将初步预测按 起始和结束对数的和 降序排序。

        _NbestPrediction = collections.namedtuple(
            "NbestPrediction", ["text", "start_logit", "end_logit"]) # 最终的最佳预测信息
        seen_predictions = {}

        # 生成最佳候选答案 nbest
        nbest = []
        for pred in prelim_predictions: # 对于每一个初步预测，转换为原始文本
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)
                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        if not nbest: # 如果没有有效的初步预测，则将文本设为空字符串，并添加一个默认的预测
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest: # 将最佳候选答案添加到列表中，并确保列表中至少有一个条目（如果没有有效预测，则添加一个空文本条目）
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        # 计算概率并构建输出
        """
        这里使用的Softmax:
        Softmax 是一种激活函数，它通常用于将一组任意实数转换为归一化的概率分布; 具体地说，Softmax 函数将输入的实数（在这里是 total_scores）转换为介于 0 和 1 之间的数值，这些数值加起来等于 1

        Softmax的作用:
        **归一化分数: Softmax 确保所有候选答案的分数都被标准化为一个概率分布，这样我们就可以比较它们的相对重要性
        概率解释: total_scores 通过 Softmax 转换后得到的值 可以被解释为 每个候选答案是正确答案 的"概率" (因为和为1，所以可以被视为概率)

        "最佳候选答案的概率" 指的是 模型 认为某个答案是正确答案的可能性 (模型对每个候选答案是正确答案的置信度)
        """
        probs = _compute_softmax(total_scores) # 使用softmax函数计算每个最佳候选答案的概率
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            # 对于每个最佳候选答案，构建包含文本、概率、起始对数和结束对数的有序字典，并将这些字典添加到nbest_json列表中。
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        # 将每个示例的最佳答案（nbest_json列表中的第一个条目）的文本存储到all_predictions字典中，并将完整的nbest_json列表存储到all_nbest_json字典中，使用示例的qas_id作为键
        assert len(nbest_json) >= 1
        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json 
    # all_predictions（包含每个示例的最佳答案文本）和all_nbest_json（包含每个示例的最佳候选答案的详细信息）


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """
    Project the tokenized prediction back to the original text.
    
    主要目的是将一个经过分词器（tokenizer）处理后的预测文本（pred_text）映射回原始文本（orig_text）中的相应位置。这在自然语言处理（NLP）任务中很常见，特别是在文本生成或文本修改任务中，模型可能输出分词后的文本，但我们需要将这些分词后的文本转换回原始的、未经分词的文本格式。
    """

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = " ".join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def get_raw_scores(dataset, predictions):
    answers = {}
    for example in dataset:
        for qa in example['qas']:
            answers[qa['qid']] = qa['answers']
    exact_scores = {}
    f1_scores = {}
    for qid, ground_truths in answers.items():
        if qid not in predictions:
            print('Missing prediction for %s' % qid)
            continue
        prediction = predictions[qid]
        exact_scores[qid] = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1_scores[qid] = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
    return exact_scores, f1_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def evaluate(args, model, device, eval_dataset, eval_dataloader,
             eval_examples, eval_features, verbose=True):
    all_results = []
    model.eval()
    for input_ids, input_mask, segment_ids, example_indices in eval_dataloader:
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    preds, nbest_preds = \
        make_predictions(eval_examples, eval_features, all_results,
                         args.n_best_size, args.max_answer_length,
                         args.do_lower_case, args.verbose_logging)
    exact_raw, f1_raw = get_raw_scores(eval_dataset, preds)
    result = make_eval_dict(exact_raw, f1_raw)
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return result, preds, nbest_preds


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = \
        args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        assert (args.train_file is not None) and (args.dev_file is not None)

    if args.eval_test:
        assert args.test_file is not None
    else:
        assert args.dev_file is not None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)

    tokenizer = BertTokenizer.from_pretrained(
        args.model, do_lower_case=args.do_lower_case)

    if args.do_train or (not args.eval_test):
        with gzip.GzipFile(args.dev_file, 'r') as reader:
            content = reader.read().decode('utf-8').strip().split('\n')[1:]
            eval_dataset = [json.loads(line) for line in content]
        eval_examples = read_mrqa_examples(
            input_file=args.dev_file, is_training=False)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)
        logger.info("***** Dev *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

    if args.do_train:
        train_examples = read_mrqa_examples(
            input_file=args.train_file, is_training=True)

        train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True)

        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = \
            len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        logger.info("***** Train *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        best_result = None
        lrs = [args.learning_rate] if args.learning_rate else [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
        for lr in lrs:
            model = BertForQuestionAnswering.from_pretrained(
                args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
            if args.fp16:
                model.half()
            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
            param_optimizer = list(model.named_parameters())
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            if args.fp16:
                try:
                    from apex.optimizers import FP16_Optimizer
                    from apex.optimizers import FusedAdam
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                      "to use distributed and fp16 training.")
                optimizer = FusedAdam(optimizer_grouped_parameters,
                                      lr=lr,
                                      bias_correction=False,
                                      max_grad_norm=1.0)
                if args.loss_scale == 0:
                    optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                else:
                    optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            else:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=lr,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            global_step = 0
            start_time = time.time()
            for epoch in range(int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                for step, batch in enumerate(train_batches):
                    if n_gpu == 1:
                        batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                    loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
                    if n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            lr_this_step = lr * \
                                warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                    if (step + 1) % eval_step == 0:
                        logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                            epoch, step + 1, len(train_dataloader), time.time() - start_time, tr_loss / nb_tr_steps))

                        save_model = False
                        if args.do_eval:
                            result, _, _ = \
                                evaluate(args, model, device, eval_dataset,
                                         eval_dataloader, eval_examples, eval_features)
                            model.train()
                            result['global_step'] = global_step
                            result['epoch'] = epoch
                            result['learning_rate'] = lr
                            result['batch_size'] = args.train_batch_size
                            if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                                best_result = result
                                save_model = True
                                logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                            (args.eval_metric, str(lr), epoch, result[args.eval_metric]))
                        else:
                            save_model = True
                        if save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(args.output_dir)
                            if best_result:
                                with open(os.path.join(args.output_dir, EVAL_FILE), "w") as writer:
                                    for key in sorted(best_result.keys()):
                                        writer.write("%s = %s\n" % (key, str(best_result[key])))

    if args.do_eval:
        if args.eval_test:
            with gzip.GzipFile(args.test_file, 'r') as reader:
                content = reader.read().decode('utf-8').strip().split('\n')[1:]
                eval_dataset = [json.loads(line) for line in content]
            eval_examples = read_mrqa_examples(
                input_file=args.test_file, is_training=False)
            eval_features = convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False)
            logger.info("***** Test *****")
            logger.info("  Num orig examples = %d", len(eval_examples))
            logger.info("  Num split examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
            eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        model = BertForQuestionAnswering.from_pretrained(args.output_dir)
        if args.fp16:
            model.half()
        model.to(device)
        result, preds, nbest_preds = \
            evaluate(args, model, device, eval_dataset,
                     eval_dataloader, eval_examples, eval_features)
        with open(os.path.join(args.output_dir, PRED_FILE), "w") as writer:
            writer.write(json.dumps(preds, indent=4) + "\n")
        with open(os.path.join(args.output_dir, TEST_FILE), "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default=None, type=str, required=True)
        parser.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output directory where the model checkpoints and predictions will be written.")
        parser.add_argument("--train_file", default=None, type=str)
        parser.add_argument("--dev_file", default=None, type=str)
        parser.add_argument("--test_file", default=None, type=str)
        parser.add_argument("--eval_per_epoch", default=10, type=int,
                            help="How many times it evaluates on dev set per epoch")
        parser.add_argument("--max_seq_length", default=384, type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                                 "longer than this will be truncated, and sequences shorter than this will be padded.")
        parser.add_argument("--doc_stride", default=128, type=int,
                            help="When splitting up a long document into chunks, "
                                 "how much stride to take between chunks.")
        parser.add_argument("--max_query_length", default=64, type=int,
                            help="The maximum number of tokens for the question. Questions longer than this will "
                                 "be truncated to this length.")
        parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
        parser.add_argument("--train_mode", type=str, default='random_sorted', choices=['random', 'sorted', 'random_sorted'])
        parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
        parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
        parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
        parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
        parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for predictions.")
        parser.add_argument("--learning_rate", default=None, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs", default=3.0, type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--eval_metric", default='f1', type=str)
        parser.add_argument("--warmup_proportion", default=0.1, type=float,
                            help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                                 "of training.")
        parser.add_argument("--n_best_size", default=20, type=int,
                            help="The total number of n-best predictions to generate in the nbest_predictions.json "
                                 "output file.")
        parser.add_argument("--max_answer_length", default=30, type=int,
                            help="The maximum length of an answer that can be generated. "
                                 "This is needed because the start "
                                 "and end predictions are not conditioned on one another.")
        parser.add_argument("--verbose_logging", action='store_true',
                            help="If true, all of the warnings related to data processing will be printed. "
                                 "A number of warnings are expected for a normal MRQA evaluation.")
        parser.add_argument("--no_cuda", action='store_true',
                            help="Whether not to use CUDA when available")
        parser.add_argument('--seed', type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--fp16', action='store_true',
                            help="Whether to use 16-bit float precision instead of 32-bit")
        parser.add_argument('--loss_scale', type=float, default=0,
                            help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                                 "0 (default value): dynamic loss scaling.\n"
                                 "Positive power of 2: static loss scaling value.\n")
        args = parser.parse_args()

        main(args)
